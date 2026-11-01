// Real Chromium verification of the embedded GLB explorer, using local CDP.
import {writeFileSync} from 'node:fs';
const args=Object.fromEntries(process.argv.slice(2).reduce((p,v,i,a)=>v.startsWith('--')?[...p,[v.slice(2),a[i+1]]]:p,[]));
const pages=await(await fetch(`http://127.0.0.1:${args.port||9227}/json/list`)).json();
const page=pages.find(p=>p.type==='page');if(!page)throw Error('No controlled Chromium page');
const ws=new WebSocket(page.webSocketDebuggerUrl);await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j});
let sequence=0;const pending=new Map(),exceptions=[],requests=[];
ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p?.reject(Error(JSON.stringify(m.error))):p?.resolve(m.result)}
 if(m.method==='Runtime.exceptionThrown')exceptions.push(m.params.exceptionDetails.text);
 if(m.method==='Network.requestWillBeSent'&&!m.params.request.url.startsWith('data:'))requests.push(m.params.request.url)};
function command(method,params={}){return new Promise((resolve,reject)=>{const id=++sequence;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}))})}
async function value(expression){const r=await command('Runtime.evaluate',{expression,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value}
async function waitFor(expression){for(let i=0;i<300;i++){if(await value(expression))return;await new Promise(r=>setTimeout(r,100))}throw Error('3D browser readiness failed: '+expression)}
const report={actual_chromium:true,actual_webgl_models:true,models:[],checks:[]};
try{
 await command('Page.enable');await command('Runtime.enable');await command('Network.enable');
 await command('Emulation.setDeviceMetricsOverride',{width:1600,height:1000,deviceScaleFactor:1,mobile:false});
 await command('Page.navigate',{url:args.url});
 await waitFor("document.getElementById('model')?.loaded && !!document.getElementById('model')?.model");
 await value("window.modelLoads=1;document.getElementById('model').addEventListener('load',()=>window.modelLoads++);document.getElementById('model').autoRotate=false");
 for(const [i,name] of ['warehouse','architecture','diversity'].entries()){
  if(i){await value(`document.querySelector('button[data-model="${name}"]').click()`);await waitFor(`window.modelLoads === ${i+1} && document.getElementById('model').loaded`)}
  const facts=await value("(()=>{const m=document.getElementById('model'),d=m.getDimensions();return {materials:m.model.materials.length,dimensions:[d.x,d.y,d.z],loaded:m.loaded,alt:m.alt}})()");
  if(facts.materials<4||facts.dimensions.some(v=>!Number.isFinite(v)||v<=0))throw Error('Incomplete model: '+name);
  report.models.push({name,...facts});
 }
 // Exercise the real camera with a pointer drag, rather than only setting a property.
 const before=await value("document.getElementById('model').getCameraOrbit().theta");
 await command('Input.dispatchMouseEvent',{type:'mouseMoved',x:810,y:530});
 await command('Input.dispatchMouseEvent',{type:'mousePressed',x:810,y:530,button:'left',clickCount:1});
 for(let i=1;i<=8;i++)await command('Input.dispatchMouseEvent',{type:'mouseMoved',x:810+i*22,y:530,button:'left',buttons:1});
 await command('Input.dispatchMouseEvent',{type:'mouseReleased',x:986,y:530,button:'left',clickCount:1});
 await waitFor(`Math.abs(document.getElementById('model').getCameraOrbit().theta - ${before}) > .05`);
 report.camera_drag_changed_orbit=true;
 await value("document.querySelector('button[data-model=architecture]').click()");await waitFor('window.modelLoads === 4');
 await value("document.getElementById('model').cameraOrbit='35deg 55deg auto';document.getElementById('model').jumpCameraToGoal()");
 const capture=await command('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});writeFileSync(args.out,Buffer.from(capture.data,'base64'));
 report.checks=['All three embedded GLBs render','Materials and model dimensions are nonempty','Model selection buttons','Pointer drag rotates actual camera','No external asset dependencies'];
 report.external_requests=requests.filter(u=>!u.startsWith(new URL(args.url).origin));
 if(exceptions.length)throw Error('Page JavaScript exceptions: '+JSON.stringify(exceptions));
 if(report.external_requests.length)throw Error('Unexpected external asset requests: '+JSON.stringify(report.external_requests));
 writeFileSync(args.report,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
}finally{ws.close()}
