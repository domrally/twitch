// #version 300 es

// Done:
// - subsurface scattering
// - fresnel
// - diffuse
// - specular
// - roughness
// - florescence
// - iridescence
// - primitive toon shader

// TODO: 
// 1) cross hatching shadows
// 2) ambient occlusion
// 3) phosphorescence (unreal)
// 4) turn subsurface scattering into transmission
// 0) iridescence
// 0) florescence

// audience polling:
// ) music level?
// ) music choice???
// ) 


#extension GL_OES_standard_derivatives : enable

#ifdef GL_ES
precision mediump float;
#endif

// mathematical constant
const float pi = 3.1415926535897932384626433832795;

// Copyright(c) 2021 Björn Ottosson
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this softwareand associated documentation files(the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and /or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions :
// The above copyright noticeand this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#define M_PI 3.1415926535897932384626433832795

float cbrt( float x )
{
    return sign(x)*pow(abs(x),1.0/3.0);
}

float srgb_transfer_function(float a)
{
	return .0031308 >= a ? 12.92 * a : 1.055 * pow(a, .4166666666666667) - .055;
}

float srgb_transfer_function_inv(float a)
{
	return .04045 < a ? pow((a + .055) / 1.055, 2.4) : a / 12.92;
}

vec3 linear_srgb_to_oklab(vec3 c)
{
	float l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
	float m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
	float s = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b;

	float l_ = cbrt(l);
	float m_ = cbrt(m);
	float s_ = cbrt(s);

	return vec3(
		0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
		1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
		0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
	);
}

vec3 oklab_to_linear_srgb(vec3 c)
{
	float l_ = c.x + 0.3963377774 * c.y + 0.2158037573 * c.z;
	float m_ = c.x - 0.1055613458 * c.y - 0.0638541728 * c.z;
	float s_ = c.x - 0.0894841775 * c.y - 1.2914855480 * c.z;

	float l = l_ * l_ * l_;
	float m = m_ * m_ * m_;
	float s = s_ * s_ * s_;

	return vec3(
		+4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
		-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
		-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
	);
}

// Finds the maximum saturation possible for a given hue that fits in sRGB
// Saturation here is defined as S = C/L
// a and b must be normalized so a^2 + b^2 == 1
float compute_max_saturation(float a, float b)
{
	// Max saturation will be when one of r, g or b goes below zero.

	// Select different coefficients depending on which component goes below zero first
	float k0, k1, k2, k3, k4, wl, wm, ws;

	if (-1.88170328 * a - 0.80936493 * b > 1.)
	{
		// Red component
		k0 = +1.19086277; k1 = +1.76576728; k2 = +0.59662641; k3 = +0.75515197; k4 = +0.56771245;
		wl = +4.0767416621; wm = -3.3077115913; ws = +0.2309699292;
	}
	else if (1.81444104 * a - 1.19445276 * b > 1.)
	{
		// Green component
		k0 = +0.73956515; k1 = -0.45954404; k2 = +0.08285427; k3 = +0.12541070; k4 = +0.14503204;
		wl = -1.2684380046; wm = +2.6097574011; ws = -0.3413193965;
	}
	else
	{
		// Blue component
		k0 = +1.35733652; k1 = -0.00915799; k2 = -1.15130210; k3 = -0.50559606; k4 = +0.00692167;
		wl = -0.0041960863; wm = -0.7034186147; ws = +1.7076147010;
	}

	// Approximate max saturation using a polynomial:
	float S = k0 + k1 * a + k2 * b + k3 * a * a + k4 * a * b;

	// Do one step Halley's method to get closer
	// this gives an error less than 10e6, except for some blue hues where the dS/dh is close to infinite
	// this should be sufficient for most applications, otherwise do two/three steps 

	float k_l = +0.3963377774 * a + 0.2158037573 * b;
	float k_m = -0.1055613458 * a - 0.0638541728 * b;
	float k_s = -0.0894841775 * a - 1.2914855480 * b;

	{
		float l_ = 1. + S * k_l;
		float m_ = 1. + S * k_m;
		float s_ = 1. + S * k_s;

		float l = l_ * l_ * l_;
		float m = m_ * m_ * m_;
		float s = s_ * s_ * s_;

		float l_dS = 3. * k_l * l_ * l_;
		float m_dS = 3. * k_m * m_ * m_;
		float s_dS = 3. * k_s * s_ * s_;

		float l_dS2 = 6. * k_l * k_l * l_;
		float m_dS2 = 6. * k_m * k_m * m_;
		float s_dS2 = 6. * k_s * k_s * s_;

		float f = wl * l + wm * m + ws * s;
		float f1 = wl * l_dS + wm * m_dS + ws * s_dS;
		float f2 = wl * l_dS2 + wm * m_dS2 + ws * s_dS2;

		S = S - f * f1 / (f1 * f1 - 0.5 * f * f2);
	}

	return S;
}

// finds L_cusp and C_cusp for a given hue
// a and b must be normalized so a^2 + b^2 == 1
vec2 find_cusp(float a, float b)
{
	// First, find the maximum saturation (saturation S = C/L)
	float S_cusp = compute_max_saturation(a, b);

	// Convert to linear sRGB to find the first point where at least one of r,g or b >= 1:
	vec3 rgb_at_max = oklab_to_linear_srgb(vec3( 1, S_cusp * a, S_cusp * b ));
	float L_cusp = cbrt(1. / max(max(rgb_at_max.r, rgb_at_max.g), rgb_at_max.b));
	float C_cusp = L_cusp * S_cusp;

	return vec2( L_cusp , C_cusp );
}

// Finds intersection of the line defined by 
// L = L0 * (1 - t) + t * L1;
// C = t * C1;
// a and b must be normalized so a^2 + b^2 == 1
float find_gamut_intersection(float a, float b, float L1, float C1, float L0, vec2 cusp)
{
	// Find the intersection for upper and lower half seprately
	float t;
	if (((L1 - L0) * cusp.y - (cusp.x - L0) * C1) <= 0.)
	{
		// Lower half

		t = cusp.y * L0 / (C1 * cusp.x + cusp.y * (L0 - L1));
	}
	else
	{
		// Upper half

		// First intersect with triangle
		t = cusp.y * (L0 - 1.) / (C1 * (cusp.x - 1.) + cusp.y * (L0 - L1));

		// Then one step Halley's method
		{
			float dL = L1 - L0;
			float dC = C1;

			float k_l = +0.3963377774 * a + 0.2158037573 * b;
			float k_m = -0.1055613458 * a - 0.0638541728 * b;
			float k_s = -0.0894841775 * a - 1.2914855480 * b;

			float l_dt = dL + dC * k_l;
			float m_dt = dL + dC * k_m;
			float s_dt = dL + dC * k_s;


			// If higher accuracy is required, 2 or 3 iterations of the following block can be used:
			{
				float L = L0 * (1. - t) + t * L1;
				float C = t * C1;

				float l_ = L + C * k_l;
				float m_ = L + C * k_m;
				float s_ = L + C * k_s;

				float l = l_ * l_ * l_;
				float m = m_ * m_ * m_;
				float s = s_ * s_ * s_;

				float ldt = 3. * l_dt * l_ * l_;
				float mdt = 3. * m_dt * m_ * m_;
				float sdt = 3. * s_dt * s_ * s_;

				float ldt2 = 6. * l_dt * l_dt * l_;
				float mdt2 = 6. * m_dt * m_dt * m_;
				float sdt2 = 6. * s_dt * s_dt * s_;

				float r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s - 1.;
				float r1 = 4.0767416621 * ldt - 3.3077115913 * mdt + 0.2309699292 * sdt;
				float r2 = 4.0767416621 * ldt2 - 3.3077115913 * mdt2 + 0.2309699292 * sdt2;

				float u_r = r1 / (r1 * r1 - 0.5 * r * r2);
				float t_r = -r * u_r;

				float g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s - 1.;
				float g1 = -1.2684380046 * ldt + 2.6097574011 * mdt - 0.3413193965 * sdt;
				float g2 = -1.2684380046 * ldt2 + 2.6097574011 * mdt2 - 0.3413193965 * sdt2;

				float u_g = g1 / (g1 * g1 - 0.5 * g * g2);
				float t_g = -g * u_g;

				float b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s - 1.;
				float b1 = -0.0041960863 * ldt - 0.7034186147 * mdt + 1.7076147010 * sdt;
				float b2 = -0.0041960863 * ldt2 - 0.7034186147 * mdt2 + 1.7076147010 * sdt2;

				float u_b = b1 / (b1 * b1 - 0.5 * b * b2);
				float t_b = -b * u_b;

				t_r = u_r >= 0. ? t_r : 10000.;
				t_g = u_g >= 0. ? t_g : 10000.;
				t_b = u_b >= 0. ? t_b : 10000.;

				t += min(t_r, min(t_g, t_b));
			}
		}
	}

	return t;
}

float find_gamut_intersection(float a, float b, float L1, float C1, float L0)
{
	// Find the cusp of the gamut triangle
	vec2 cusp = find_cusp(a, b);

	return find_gamut_intersection(a, b, L1, C1, L0, cusp);
}

vec3 gamut_clip_preserve_chroma(vec3 rgb)
{
	if (rgb.r < 1. && rgb.g < 1. && rgb.b < 1. && rgb.r > 0. && rgb.g > 0. && rgb.b > 0.)
		return rgb;

	vec3 lab = linear_srgb_to_oklab(rgb);

	float L = lab.x;
	float eps = 0.00001;
	float C = max(eps, sqrt(lab.y * lab.y + lab.z * lab.z));
	float a_ = lab.y / C;
	float b_ = lab.z / C;

	float L0 = clamp(L, 0., 1.);

	float t = find_gamut_intersection(a_, b_, L, C, L0);
	float L_clipped = L0 * (1. - t) + t * L;
	float C_clipped = t * C;

	return oklab_to_linear_srgb(vec3( L_clipped, C_clipped * a_, C_clipped * b_ ));
}

vec3 gamut_clip_project_to_0_5(vec3 rgb)
{
	if (rgb.r < 1. && rgb.g < 1. && rgb.b < 1. && rgb.r > 0. && rgb.g > 0. && rgb.b > 0.)
		return rgb;

	vec3 lab = linear_srgb_to_oklab(rgb);

	float L = lab.x;
	float eps = 0.00001;
	float C = max(eps, sqrt(lab.y * lab.y + lab.z * lab.z));
	float a_ = lab.y / C;
	float b_ = lab.z / C;

	float L0 = 0.5;

	float t = find_gamut_intersection(a_, b_, L, C, L0);
	float L_clipped = L0 * (1. - t) + t * L;
	float C_clipped = t * C;

	return oklab_to_linear_srgb(vec3( L_clipped, C_clipped * a_, C_clipped * b_ ));
}

vec3 gamut_clip_project_to_L_cusp(vec3 rgb)
{
	if (rgb.r < 1. && rgb.g < 1. && rgb.b < 1. && rgb.r > 0. && rgb.g > 0. && rgb.b > 0.)
		return rgb;

	vec3 lab = linear_srgb_to_oklab(rgb);

	float L = lab.x;
	float eps = 0.00001;
	float C = max(eps, sqrt(lab.y * lab.y + lab.z * lab.z));
	float a_ = lab.y / C;
	float b_ = lab.z / C;

	// The cusp is computed here and in find_gamut_intersection, an optimized solution would only compute it once.
	vec2 cusp = find_cusp(a_, b_);

	float L0 = cusp.x;

	float t = find_gamut_intersection(a_, b_, L, C, L0);

	float L_clipped = L0 * (1. - t) + t * L;
	float C_clipped = t * C;

	return oklab_to_linear_srgb(vec3( L_clipped, C_clipped * a_, C_clipped * b_ ));
}

vec3 gamut_clip_adaptive_L0_0_5(vec3 rgb, float alpha)
{
	if (rgb.r < 1. && rgb.g < 1. && rgb.b < 1. && rgb.r > 0. && rgb.g > 0. && rgb.b > 0.)
		return rgb;

	vec3 lab = linear_srgb_to_oklab(rgb);

	float L = lab.x;
	float eps = 0.00001;
	float C = max(eps, sqrt(lab.y * lab.y + lab.z * lab.z));
	float a_ = lab.y / C;
	float b_ = lab.z / C;

	float Ld = L - 0.5;
	float e1 = 0.5 + abs(Ld) + alpha * C;
	float L0 = 0.5 * (1. + sign(Ld) * (e1 - sqrt(e1 * e1 - 2. * abs(Ld))));

	float t = find_gamut_intersection(a_, b_, L, C, L0);
	float L_clipped = L0 * (1. - t) + t * L;
	float C_clipped = t * C;

	return oklab_to_linear_srgb(vec3( L_clipped, C_clipped * a_, C_clipped * b_ ));
}

vec3 gamut_clip_adaptive_L0_L_cusp(vec3 rgb, float alpha)
{
	if (rgb.r < 1. && rgb.g < 1. && rgb.b < 1. && rgb.r > 0. && rgb.g > 0. && rgb.b > 0.)
		return rgb;

	vec3 lab = linear_srgb_to_oklab(rgb);

	float L = lab.x;
	float eps = 0.00001;
	float C = max(eps, sqrt(lab.y * lab.y + lab.z * lab.z));
	float a_ = lab.y / C;
	float b_ = lab.z / C;

	// The cusp is computed here and in find_gamut_intersection, an optimized solution would only compute it once.
	vec2 cusp = find_cusp(a_, b_);

	float Ld = L - cusp.x;
	float k = 2. * (Ld > 0. ? 1. - cusp.x : cusp.x);

	float e1 = 0.5 * k + abs(Ld) + alpha * C / k;
	float L0 = cusp.x + 0.5 * (sign(Ld) * (e1 - sqrt(e1 * e1 - 2. * k * abs(Ld))));

	float t = find_gamut_intersection(a_, b_, L, C, L0);
	float L_clipped = L0 * (1. - t) + t * L;
	float C_clipped = t * C;

	return oklab_to_linear_srgb(vec3( L_clipped, C_clipped * a_, C_clipped * b_ ));
}

float toe(float x)
{
	float k_1 = 0.206;
	float k_2 = 0.03;
	float k_3 = (1. + k_1) / (1. + k_2);
	return 0.5 * (k_3 * x - k_1 + sqrt((k_3 * x - k_1) * (k_3 * x - k_1) + 4. * k_2 * k_3 * x));
}

float toe_inv(float x)
{
	float k_1 = 0.206;
	float k_2 = 0.03;
	float k_3 = (1. + k_1) / (1. + k_2);
	return (x * x + k_1 * x) / (k_3 * (x + k_2));
}

vec2 to_ST(vec2 cusp)
{
	float L = cusp.x;
	float C = cusp.y;
	return vec2( C / L, C / (1. - L) );
}

// Returns a smooth approximation of the location of the cusp
// This polynomial was created by an optimization process
// It has been designed so that S_mid < S_max and T_mid < T_max
vec2 get_ST_mid(float a_, float b_)
{
	float S = 0.11516993 + 1. / (
		+7.44778970 + 4.15901240 * b_
		+ a_ * (-2.19557347 + 1.75198401 * b_
			+ a_ * (-2.13704948 - 10.02301043 * b_
				+ a_ * (-4.24894561 + 5.38770819 * b_ + 4.69891013 * a_
					)))
		);

	float T = 0.11239642 + 1. / (
		+1.61320320 - 0.68124379 * b_
		+ a_ * (+0.40370612 + 0.90148123 * b_
			+ a_ * (-0.27087943 + 0.61223990 * b_
				+ a_ * (+0.00299215 - 0.45399568 * b_ - 0.14661872 * a_
					)))
		);

	return vec2( S, T );
}

vec3 get_Cs(float L, float a_, float b_)
{
	vec2 cusp = find_cusp(a_, b_);

	float C_max = find_gamut_intersection(a_, b_, L, 1., L, cusp);
	vec2 ST_max = to_ST(cusp);
	
	// Scale factor to compensate for the curved part of gamut shape:
	float k = C_max / min((L * ST_max.x), (1. - L) * ST_max.y);

	float C_mid;
	{
		vec2 ST_mid = get_ST_mid(a_, b_);

		// Use a soft minimum function, instead of a sharp triangle shape to get a smooth value for chroma.
		float C_a = L * ST_mid.x;
		float C_b = (1. - L) * ST_mid.y;
		C_mid = 0.9 * k * sqrt(sqrt(1. / (1. / (C_a * C_a * C_a * C_a) + 1. / (C_b * C_b * C_b * C_b))));
	}

	float C_0;
	{
		// for C_0, the shape is independent of hue, so vec2 are constant. Values picked to roughly be the average values of vec2.
		float C_a = L * 0.4;
		float C_b = (1. - L) * 0.8;

		// Use a soft minimum function, instead of a sharp triangle shape to get a smooth value for chroma.
		C_0 = sqrt(1. / (1. / (C_a * C_a) + 1. / (C_b * C_b)));
	}

	return vec3( C_0, C_mid, C_max );
}

vec3 okhsl_to_srgb(vec3 hsl)
{
	float h = hsl.x;
	float s = hsl.y;
	float l = hsl.z;

	if (l == 1.0)
	{
		return vec3( 1., 1., 1. );
	}

	else if (l == 0.)
	{
		return vec3( 0., 0., 0. );
	}

	float a_ = cos(2. * M_PI * h);
	float b_ = sin(2. * M_PI * h);
	float L = toe_inv(l);

	vec3 cs = get_Cs(L, a_, b_);
	float C_0 = cs.x;
	float C_mid = cs.y;
	float C_max = cs.z;

	float mid = 0.8;
	float mid_inv = 1.25;

	float C, t, k_0, k_1, k_2;

	if (s < mid)
	{
		t = mid_inv * s;

		k_1 = mid * C_0;
		k_2 = (1. - k_1 / C_mid);

		C = t * k_1 / (1. - k_2 * t);
	}
	else
	{
		t = (s - mid)/ (1. - mid);

		k_0 = C_mid;
		k_1 = (1. - mid) * C_mid * C_mid * mid_inv * mid_inv / C_0;
		k_2 = (1. - (k_1) / (C_max - C_mid));

		C = k_0 + t * k_1 / (1. - k_2 * t);
	}

	vec3 rgb = oklab_to_linear_srgb(vec3( L, C * a_, C * b_ ));
	return vec3(
		srgb_transfer_function(rgb.r),
		srgb_transfer_function(rgb.g),
		srgb_transfer_function(rgb.b)
	);
}

vec3 srgb_to_okhsl(vec3 rgb)
{
	vec3 lab = linear_srgb_to_oklab(vec3(
		srgb_transfer_function_inv(rgb.r),
		srgb_transfer_function_inv(rgb.g),
		srgb_transfer_function_inv(rgb.b)
		));

	float C = sqrt(lab.y * lab.y + lab.z * lab.z);
	float a_ = lab.y / C;
	float b_ = lab.z / C;

	float L = lab.x;
	float h = 0.5 + 0.5 * atan(-lab.z, -lab.y) / M_PI;

	vec3 cs = get_Cs(L, a_, b_);
	float C_0 = cs.x;
	float C_mid = cs.y;
	float C_max = cs.z;

	// Inverse of the interpolation in okhsl_to_srgb:

	float mid = 0.8;
	float mid_inv = 1.25;

	float s;
	if (C < C_mid)
	{
		float k_1 = mid * C_0;
		float k_2 = (1. - k_1 / C_mid);

		float t = C / (k_1 + k_2 * C);
		s = t * mid;
	}
	else
	{
		float k_0 = C_mid;
		float k_1 = (1. - mid) * C_mid * C_mid * mid_inv * mid_inv / C_0;
		float k_2 = (1. - (k_1) / (C_max - C_mid));

		float t = (C - k_0) / (k_1 + k_2 * (C - k_0));
		s = mid + (1. - mid) * t;
	}

	float l = toe(L);
	return vec3( h, s, l );
}


vec3 okhsv_to_srgb(vec3 hsv)
{
	float h = hsv.x;
	float s = hsv.y;
	float v = hsv.z;

	float a_ = cos(2. * M_PI * h);
	float b_ = sin(2. * M_PI * h);
	
	vec2 cusp = find_cusp(a_, b_);
	vec2 ST_max = to_ST(cusp);
	float S_max = ST_max.x;
	float T_max = ST_max.y;
	float S_0 = 0.5;
	float k = 1.- S_0 / S_max;

	// first we compute L and V as if the gamut is a perfect triangle:

	// L, C when v==1:
	float L_v = 1.   - s * S_0 / (S_0 + T_max - T_max * k * s);
	float C_v = s * T_max * S_0 / (S_0 + T_max - T_max * k * s);

	float L = v * L_v;
	float C = v * C_v;

	// then we compensate for both toe and the curved top part of the triangle:
	float L_vt = toe_inv(L_v);
	float C_vt = C_v * L_vt / L_v;

	float L_new = toe_inv(L);
	C = C * L_new / L;
	L = L_new;

	vec3 rgb_scale = oklab_to_linear_srgb(vec3( L_vt, a_ * C_vt, b_ * C_vt ));
	float scale_L = cbrt(1. / max(max(rgb_scale.r, rgb_scale.g), max(rgb_scale.b, 0.)));

	L = L * scale_L;
	C = C * scale_L;

	vec3 rgb = oklab_to_linear_srgb(vec3( L, C * a_, C * b_ ));
	return vec3(
		srgb_transfer_function(rgb.r),
		srgb_transfer_function(rgb.g),
		srgb_transfer_function(rgb.b)
	);
}

vec3 srgb_to_okhsv(vec3 rgb)
{
	vec3 lab = linear_srgb_to_oklab(vec3(
		srgb_transfer_function_inv(rgb.r),
		srgb_transfer_function_inv(rgb.g),
		srgb_transfer_function_inv(rgb.b)
		));

	float C = sqrt(lab.y * lab.y + lab.z * lab.z);
	float a_ = lab.y / C;
	float b_ = lab.z / C;

	float L = lab.x;
	float h = 0.5 + 0.5 * atan(-lab.z, -lab.y) / M_PI;

	vec2 cusp = find_cusp(a_, b_);
	vec2 ST_max = to_ST(cusp);
	float S_max = ST_max.x;
	float T_max = ST_max.y;
	float S_0 = 0.5;
	float k = 1. - S_0 / S_max;

	// first we find L_v, C_v, L_vt and C_vt

	float t = T_max / (C + L * T_max);
	float L_v = t * L;
	float C_v = t * C;

	float L_vt = toe_inv(L_v);
	float C_vt = C_v * L_vt / L_v;

	// we can then use these to invert the step that compensates for the toe and the curved top part of the triangle:
	vec3 rgb_scale = oklab_to_linear_srgb(vec3( L_vt, a_ * C_vt, b_ * C_vt ));
	float scale_L = cbrt(1. / max(max(rgb_scale.r, rgb_scale.g), max(rgb_scale.b, 0.)));

	L = L / scale_L;
	C = C / scale_L;

	C = C * toe(L) / L;
	L = toe(L);

	// we can now compute v and s:

	float v = L / L_v;
	float s = (S_0 + T_max) * C_v / ((T_max * S_0) + T_max * k * C_v);

	return vec3 (h, s, v );
}

// BEGIN CUSTOM SHADER CODE

struct Light
{
    float intensity;
    vec3 position;
    vec3 color;
};

// distribution of reflective microfacets on a surface
// see: https://en.wikipedia.org/wiki/Specular_highlight#Beckmann_distribution
float beckmannDistribution(float HN)
{
    // roughness
    const float m = 0.5;
    const float m2 = m * m;
    
    float cos2 = HN * HN;
    float cos4 = cos2 * cos2;
    
    float num = exp((cos2 - 1.0) / (cos2 * m2));
    float den = pi * m2 * cos4;
    
    float k = num / den;
    
    return k;
}

// amount of light reflected at boundary of air and object 
// see: https://en.wikipedia.org/wiki/Schlick%27s_approximation
float schlickApproximation(float VN)
{
    /// refractive indices
    // air
    const float n_1 = 1.000293;
    // diamond
    const float n_2 = 2.417;
    
    float R_0 = (n_1 - n_2) / (n_1 + n_2);
    R_0 *= R_0;
    
    float incidence = 1.0 - VN;
    incidence *= incidence * incidence * incidence * incidence;
    float R = R_0 + (1.0 - R_0) * incidence;
    
    return R;
}

// specular reflections off a rough surface
// see: https://en.wikipedia.org/wiki/Specular_highlight#Cook%E2%80%93Torrance_model
vec3 cookTorrance(vec3 V, vec3 N, Light L)
{
    vec3 H = normalize(V + L.position);
    float VN = dot(V, N);
    float NL = dot(N, L.position);
    float HN = dot(H, N);
    float VH = dot(V, H);
    
    float g = 2.0 * HN / VH;
    float G = 1.0;
    G = min(G, g * VN);
    G = min(G, g * NL);
    
    float D = beckmannDistribution(HN);
    float F = schlickApproximation(VN);
    
    float k = 0.25 * D * F * G / (VN * NL);
    
    vec3 color = k * L.intensity * L.color;

	//  color = .25 + .5 * color;
	 color *= .5;

	 color = clamp(color, 0.0, 1.0);
        
    return color;
}

// diffuse reflections on a rough surface
// see: https://en.wikipedia.org/wiki/Oren%E2%80%93Nayar_reflectance_model
float orenNayar(vec3 normal, Light light, vec3 eye)
{
    // roughness
    const float sigma = 0.3;
    const float sigma2 = sigma * sigma;
    const float A = 1.0 - 0.5 * sigma2 / (sigma2 + 0.33);
    const float B = 0.45 * sigma2 / (sigma2 + 0.09);
    
    float eyeNormalProjection = dot(eye, normal);
    float lightNormalProjection = dot(light.position, normal);
    float thetaEye = acos(eyeNormalProjection);
    float thetaLight = acos(lightNormalProjection);
    float alpha = max(thetaEye, thetaLight);
    float beta = min(thetaEye, thetaLight);
    vec3 eyeNormalRejection = eye - eyeNormalProjection * normal;
    vec3 lightNormalRejection = eye - lightNormalProjection * normal;
    vec3 axis = cross(vec3(0.0, 1.0, 0.0), normal);
    float phiEye = acos(dot(eyeNormalRejection, axis));
    float phiLight = acos(dot(lightNormalRejection, axis));
    
    float L = A + B * max(cos(phiEye - phiLight), 0.0) * sin(alpha) * tan(beta);
    L *= max(lightNormalProjection, 0.0);
    L *= light.intensity;

	//  L = .25 + .5 * L;
	 L = clamp(L, 0.0, 1.0);
    
    return L;
}

// scattering of light below the surface of the object
float sss(Light light, float orenNayar)
{
    const float scatterWidth = 0.3;
    const float scatterWidth2 = 2.0 * scatterWidth;
    float s = light.intensity * smoothstep(0.0, scatterWidth, orenNayar) * smoothstep(scatterWidth2, scatterWidth, orenNayar);
        
    return s;
}

varying vec4 v_depth;
varying vec4 v_position;
varying vec4 v_normal;
varying vec2 v_texcoord;
varying vec4 v_color;

uniform mat4 u_projectionMatrix;
uniform mat4 u_modelViewMatrix;
uniform mat4 u_normalMatrix;
uniform vec2 u_resolution;
uniform float u_time;

#if defined(VERTEX)

// attribute vec4 a_position; // myfolder/myfile.obj
attribute vec4 a_position;
attribute vec4 a_normal;
attribute vec2 a_texcoord;
attribute vec4 a_color;

void main(void) {
	v_position = u_projectionMatrix * u_modelViewMatrix * a_position;
	v_normal = u_normalMatrix * a_normal;
	v_depth = u_projectionMatrix * u_modelViewMatrix * a_position;
	v_texcoord = a_texcoord;// + vec2(0.5, 0.5) + sin(u_time + a_position.x + a_position.y + a_position.z) ;
	v_color = a_color;
	gl_Position = v_position;// + v_normal * .2 * (0.5 + .5 * sin(u_time + v_position.x + v_position.y + v_position.z));
}

#else // fragment shaderxas
// it's the end of the thing
uniform vec2 u_mouse;
uniform vec2 u_pos;
// uniform sampler2D u_texture; // https://cdn.jsdelivr.net/gh/actarian/plausible-brdf-shader/textures/mars/4096x2048/diffuse.jpg?repeat=true
// uniform vec2 u_textureResolution;

float checker(vec2 uv, float repeats) {
	float cx = floor(repeats * uv.x);
	float cy = floor(repeats * uv.y);
	float result = mod(cx + cy, 2.0);
	return sign(result);
}

void main() {
	vec2 p = v_texcoord;

	float numberOfLevels = 2.0;

	// rembrandt lighting
	Light bulb = Light(2., vec3(100.0, 100.0, -100.0), vec3(1.0, 1.0, 1.0));
	Light bulv = Light(1., normalize(vec3(-100.0, 100.0, -100.0)), vec3(1.0, 1.0, 1.0));

   // checkerboard pattern
	vec3 checkerBoard = vec3(checker(p, 8.0));
	// vec3 color =vec3(.3, .3, 0.7);
	vec3 color =vec3(.3, .3, .3);
	vec3 floresce =vec3(1.0, 0.9686, 0.502);
	float e = dot(v_normal.xyz, vec3(0.0, 0.0, 1.) );
	vec3 edge = vec3(floor(2.* abs(pow(e, .7))));

	// iridescent highlights
	vec3 iridescent = okhsl_to_srgb(vec3(10. * (0.5 + 0.5 * (dot(v_normal.xyz, normalize(bulb.position)))), 1., .5) );
	vec3 iridescent2 = okhsl_to_srgb(vec3(10. * (0.5 + 0.5 * (dot(v_normal.xyz, normalize(bulv.position)))), 1., .5) );

	//  specular highlights and reflections
	vec3 roughSpecular = iridescent  * cookTorrance(v_position.xyz, v_normal.xyz, bulb);
	vec3 roughSpecular2 = iridescent2  * cookTorrance(v_position.xyz, v_normal.xyz, bulv);

	//  diffuse reflections
	float d =  orenNayar(v_normal.xyz, bulb, vec3(0.0,0.0, -1.0));
	float d2 =   orenNayar(v_normal.xyz, bulv, vec3(0.0,0.0, -1.0));
	vec3 diffuse = color * d;
	vec3 diffuse2 = floresce * d2;

	// redshifted subsurface scattering
	vec3 sss1 = color * vec3(1., .75, .5) * sss(bulb, d);
	vec3 sss2 = color * vec3(1., .75, .5) * sss(bulv, d2);

	gl_FragColor = vec4(floor(numberOfLevels * (vec3(.2)+ diffuse + diffuse2 + roughSpecular + roughSpecular2+ sss1+ sss2)) * edge / numberOfLevels, 1.0);
	gl_FragColor = vec4((vec3(.4)+ diffuse + diffuse2 + roughSpecular + roughSpecular2+ sss1+ sss2) * checkerBoard, 1.0);
	gl_FragColor = vec4((vec3(.4) + diffuse + roughSpecular + sss1) + (diffuse2 + roughSpecular2+ sss2) * checkerBoard, 1.0);
	gl_FragColor = vec4((vec3(.4) + diffuse + roughSpecular + sss1) + (diffuse2 + roughSpecular2+ sss2) * checkerBoard, 1.0);
	// gl_FragColor = vec4(vec3(.4), 1.0);
	// gl_FragColor = vec4(diffuse + diffuse2, 1.0);
	// gl_FragColor = vec4((roughSpecular + roughSpecular2), 1.0);
	// gl_FragColor = vec4((iridescent +iridescent2), 1.0);
	// gl_FragColor = vec4((sss1 + sss2), 1.0);
	// gl_FragColor = vec4(floor(numberOfLevels * (diffuse + roughSpecular + sss1) + (diffuse2 + roughSpecular2+ sss2) * edge* checkerBoard / numberOfLevels), 1.0);
}

#endif
