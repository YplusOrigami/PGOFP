import fs from 'node:fs/promises';
import {importRuntimeModule} from '/Users/wuaijia/.codex/plugins/cache/openai-primary-runtime/presentations/26.1025.11240/skills/presentations/container_tools/runtime_helpers.mjs';
const {PresentationFile,FileBlob}=await importRuntimeModule('@oai/artifact-tool');
const p=await PresentationFile.importPptx(await FileBlob.load('/Volumes/T7/projects/PGOFP/output/ppt/TokamaCon_19页_封面含论文原标题.pptx'));
const b=await p.export({slide:p.slides.items[0],format:'png',scale:1});
await fs.writeFile('/Volumes/T7/projects/PGOFP/.tokamacon_ppt_build/cover_title_preview.png',new Uint8Array(await b.arrayBuffer()));
console.log('rendered cover');
