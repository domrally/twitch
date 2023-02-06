import fastifyStatic from '@fastify/static'
import * as path from 'path'
import { fileURLToPath } from 'url'
import { Fastify } from './Fastify'

/**
 * this is the server entry point
 */
export class Server {
	constructor(private port: number) {
		this.init()
	}

	async init() {
		const __filename = fileURLToPath(import.meta.url),
			__dirname = path.dirname(__filename)

		// serve static fx files
		await Fastify.register(fastifyStatic, {
			root: path.join(__dirname, 'toys'),
			prefix: '/toys/', // optional: default '/'
		})

		// wait for fastify to finish warming up, then call the ready hanler
		await Fastify.ready()
		console.log('Server is ready')

		// start the server by listening for client request
		await Fastify.listen({ port: this.port })
		console.log(`Server is listening on port ${this.port}`)
	}
}
