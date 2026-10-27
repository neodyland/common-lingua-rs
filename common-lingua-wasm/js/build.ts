import {readFile,writeFile} from "node:fs/promises"

const json = JSON.parse(await readFile("./package.json"))
delete json["devDependencies"]
delete json["scripts"]
delete json["packageManager"]
delete json["devEngines"]
await writeFile("./build/package.json", JSON.stringify(json, null ,4))
