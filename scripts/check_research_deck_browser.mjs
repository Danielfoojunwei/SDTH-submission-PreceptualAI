// Real Chromium/CDP checks. No browser framework or fabricated screenshots.
import {writeFileSync} from 'node:fs';
import {resolve} from 'node:path';
const args=Object.fromEntries(process.argv.slice(2).reduce((pairs,v,i,a)=>v.startsWith('--')?[...pairs,[v.slice(2),a[i+1]]]:pairs,[]));
const port=args.port||'9227';
const pages=await(await fetch(`http://127.0.0.1:${port}/json/list`)).json();
const page=pages.find(p=>p.type==='page');if(!page)throw Error('No controlled Chromium page');
const ws=new WebSocket(page.webSocketDebuggerUrl);await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j});
let sequence=0;const pending=new Map();ws.onmessage=e=>{const message=JSON.parse(e.data);if(message.id){const p=pending.get(message.id);pending.delete(message.id);if(message.error)p?.reject(Error(JSON.stringify(message.error)));else p?.resolve(message.result)}};
function command(method,params={}){return new Promise((resolve,reject)=>{const id=++sequence;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}))})}
async function value(expression){const r=await command('Runtime.evaluate',{expression,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value}
async function waitFor(expression){for(let i=0;i<100;i++){if(await value(expression))return;await new Promise(r=>setTimeout(r,100))}throw Error('Browser readiness failed: '+expression)}
async function key(key){await command('Input.dispatchKeyEvent',{type:'keyDown',key});await command('Input.dispatchKeyEvent',{type:'keyUp',key})}
await command('Page.enable');await command('Network.enable');await command('Network.setCacheDisabled',{cacheDisabled:true});await command('Emulation.setDeviceMetricsOverride',{width:1600,height:1000,deviceScaleFactor:1,mobile:false});
const result={actual_chromium:true,network_targets:'Local workspace artifacts only',checks:[]};
try{
 if(args.studio){
  await command('Page.navigate',{url:(args.url||('file://'+resolve(args.studio)))+'#map'});
  await waitFor("document.readyState === 'complete' && document.querySelector('#view svg') !== null");
  const capture=await command('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
  writeFileSync(args.out,Buffer.from(capture.data,'base64'));result.checks.push('Actual Studio fleet-map screenshot captured');
 }
 if(args.deck){
  await command('Page.navigate',{url:(args.url?(args.url+'?verification='+Date.now()):('file://'+resolve(args.deck)))+'#5'});
  await waitFor("document.getElementById('position')?.textContent?.startsWith('5 /') && document.getElementById('slide')?.complete && document.getElementById('slide')?.naturalWidth > 0");
  const before=await value("document.getElementById('position').textContent");
  await key('ArrowRight');await waitFor("document.getElementById('position').textContent.startsWith('6 /')");
  await key('n');await waitFor("document.getElementById('notes').style.display === 'block'");
  await key('n');await key('o');await waitFor("document.getElementById('overview').style.display === 'grid'");
  const count=await value("document.querySelectorAll('#overview img').length");await key('Escape');
  await key('ArrowLeft');await waitFor("document.getElementById('position').textContent.startsWith('5 /')");
  const capture=await command('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
  writeFileSync(args.out,Buffer.from(capture.data,'base64'));
  result.checks.push('Slide image renders','Arrow-key navigation','Notes toggle','Overview grid','Escape closes overlay');result.slide_count=count;result.initial_position=before;
 }
 if(args.report)writeFileSync(args.report,JSON.stringify(result,null,2)+'\n');
 console.log(JSON.stringify(result));
}finally{ws.close()}
