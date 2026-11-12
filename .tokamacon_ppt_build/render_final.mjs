import fs from 'node:fs/promises';
import {importRuntimeModule} from '/Users/wuaijia/.codex/plugins/cache/openai-primary-runtime/presentations/26.1024.11018/skills/presentations/container_tools/runtime_helpers.mjs';
const {PresentationFile,FileBlob}=await importRuntimeModule('@oai/artifact-tool');
const p=await PresentationFile.importPptx(await FileBlob.load('/private/tmp/tokamacon-finalize/output/ppt/TokamaCon_15页_30分钟.pptx'));
const root='/Volumes/T7/projects/PGOFP/.tokamacon_ppt_build';
for(let i=0;i<p.slides.items.length;i++){
 const b=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile(`${root}/verified-${String(i+1).padStart(2,'0')}.png`,new Uint8Array(await b.arrayBuffer()));
}
console.log(JSON.stringify({rendered:p.slides.items.length}));
