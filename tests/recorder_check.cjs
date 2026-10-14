const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync('src/ui/live_recorder_component/frontend/index.html', 'utf8');
const nodes = {}, handlers = {}, sent = [], timers = new Map();
let tick = 0, nextTimer = 0, trackStopped = false, instance;
const node = () => ({classList:{toggle(){}}, setAttribute(k,v){this[k]=v}, addEventListener(k,v){this[k]=v}});
class Recorder {
  static isTypeSupported(){return true}
  constructor(){instance=this;this.mimeType='audio/webm';this.state='inactive'}
  start(){this.state='recording'}
  pause(){this.state='paused'}
  resume(){this.state='recording'}
  stop(){this.state='inactive';this.ondataavailable({data:new Blob(['audio'])});this.onstop()}
}
class Reader {
  readAsDataURL(){this.result='data:audio/webm;base64,YXVkaW8=';this.onloadend()}
}
const window = {MediaRecorder:Recorder, parent:{postMessage(message){if(message.type==='streamlit:setComponentValue')sent.push(message.value)}},
  addEventListener(k,v){handlers[k]=v}, setTimeout(fn){timers.set(++nextTimer,fn);return nextTimer},
  clearTimeout(id){timers.delete(id)}, setInterval(){return 0}, clearInterval(){}};
const context = vm.createContext({window, document:{documentElement:{dataset:{}},getElementById(id){return nodes[id] ||= node()}},
  navigator:{mediaDevices:{async getUserMedia(){return {getTracks(){return [{stop(){trackStopped=true}}]}}}}},
  MediaRecorder:Recorder,FileReader:Reader,Blob,Date:class extends Date {static now(){return tick}},Math});
vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1], context);
const render = (args) => handlers.message({data:{type:'streamlit:render',args}});
(async()=>{
  render({note_id:'A'});
  await vm.runInContext('start()', context);
  assert.equal(instance.state,'recording');
  assert.equal(sent[0].note_id,'A');
  render({note_id:'B',ack_nonce:sent.at(-1).nonce}); // Other note/view while recording.
  assert.equal(trackStopped,false);
  tick=3000;nodes.pause.click();
  assert.equal(instance.state,'paused');
  assert.equal(nodes.pause['aria-label'],'녹음 재개');
  tick=8000;nodes.pause.click();
  assert.equal(instance.state,'recording');
  assert.equal(nodes.elapsed.textContent,'00:00:03');
  nodes.pause.click();nodes.stop.click(); // Stop is independent and works while paused.
  assert.equal(trackStopped,true);
  const audio=sent.find(item=>item.action==='audio_chunk');
  assert.equal(audio.note_id,'A');assert.equal(audio.final,true);
  render({note_id:'B',ack_nonce:audio.nonce});
  assert.equal(sent.at(-1).active,false);
  render({note_id:'B',ack_nonce:sent.at(-1).nonce});
  assert.equal(nodes.record.disabled,false);
  await vm.runInContext('start()', context);
  assert.equal(sent.at(-1).note_id,'B');
  console.log('PASS: independent pause/resume/stop, elapsed time, stable recording target and acknowledged final chunk');
})().catch(error=>{console.error(error);process.exitCode=1});
