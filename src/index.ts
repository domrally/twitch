import chalk from 'chalk'
import Color from 'colorjs.io'
import { log } from 'console'
import inquirer from 'inquirer'
import logUpdate from 'log-update'
// import Rx from 'rxjs'

// const prompts = new Rx.Subject()
let answers = await inquirer.prompt([
	{
		type: 'input',
		name: 'nameInput',
		message: 'What is your name?',
	},
])

await load()

log(`Hello ${answers.nameInput}!\n`)
await load()
answers = await inquirer.prompt({
	type: 'input',
	name: 'favColorInput',
	message: 'What is your favorite color?',
})

let customColor = new Color(answers.favColorInput),
	{ r, g, b } = customColor.toGamut({ space: 'srgb' }).srgb,
	{ l, c, h } = customColor.oklch,
	{
		r: r2,
		g: g2,
		b: b2,
	} = new Color('oklch', [l!, c!, h! + 120]).toGamut({ space: 'srgb' }).srgb

r = convert256(r)
g = convert256(g)
b = convert256(b)
r2 = convert256(r2)
g2 = convert256(g2)
b2 = convert256(b2)

await load()

log(
	`Wow, your favorite color is ${chalk.rgb(
		r!,
		g!,
		b!
	)(answers.favColorInput)}?\n`
)
await load()
log(
	'My favorite color is ' +
		chalk.rgb(r2!, g2!, b2!)('complementary to your color') +
		'!\n'
)

await load()
answers = await inquirer.prompt({
	type: 'list',
	name: 'questInput',
	choices: ['Yes, I can help you!', 'No thanks, I am very very busy.'],
	message: 'Hey, I have something I need help with, could you try to help me?',
})
await load()
log(`Oh thank you so much! I really appreciate it!\n`)

function convert256(zeroToOne?: number) {
	return Math.round(zeroToOne! * 255)
}

async function load() {
	const frames = ['-', '\\', '|', '/']
	let index = 0

	const loader = setInterval(() => {
		const frame = frames[(index = ++index % frames.length)]

		logUpdate(`${frame}`)
	}, 80)
	await new Promise(resolve => {
		setTimeout(resolve, 3000)
	})
	clearInterval(loader)
}
