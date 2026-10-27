//@ts-ignore
import wasm from "./build/raw/common_lingua_wasm_bg.wasm";
//@ts-ignore
import init, { detect as wasmDetect } from "./build/raw/common_lingua_wasm.js";

await init({ module_or_path: wasm });

export function detect(s: string): Record<string, number> {
  return Object.fromEntries(wasmDetect(s));
}
