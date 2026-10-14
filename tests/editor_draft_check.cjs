const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const storage = new Map();
const localStorage = {
  getItem: key => storage.get(key) ?? null,
  setItem: (key, value) => storage.set(key, value),
  removeItem: key => storage.delete(key),
};
const editor = fs.readFileSync('src/ui/editable_component/frontend/index.html', 'utf8');
const context = vm.createContext({localStorage, state:{draft_key:'memo-A'}});
vm.runInContext(editor.slice(editor.indexOf('    function draftKey('), editor.indexOf('    function post(')), context);
vm.runInContext('keepDraft("memo", "saved", "pending")', context);
assert.equal(vm.runInContext('draft("memo", "saved").text', context), 'pending');
assert.equal(vm.runInContext('draft("memo", "changed remotely")', context), null);
vm.runInContext('state.draft_key = "memo-B"', context);
assert.equal(vm.runInContext('draft("memo", "saved")', context), null);
vm.runInContext('state.draft_key = "memo-A"; clearDraft("memo")', context);
assert.equal(vm.runInContext('draft("memo", "saved")', context), null);
storage.set('lass-draft:memo-A:memo', '{invalid');
assert.equal(vm.runInContext('draft("memo", "saved")', context), null);

let marker = {dataset:{session:'A', revision:'message-1'}};
class Field {
  constructor() { this.dataset = {}; this.handlers = new Set(); this.text = ''; }
  get value() { return this.text; }
  set value(value) { this.text = value; }
  addEventListener(_, handler) { this.handlers.add(handler); }
  removeEventListener(_, handler) { this.handlers.delete(handler); }
  dispatchEvent() { for (const handler of this.handlers) handler(); }
}
let field = new Field();
const parentDocument = {querySelector: selector => selector === '#lass-chat-draft' ? marker : field};
const layout = fs.readFileSync('src/ui/layout_component/frontend/index.html', 'utf8');
const chatContext = vm.createContext({parentDocument, window:{parent:{localStorage, HTMLTextAreaElement:Field, Event:class {}}}});
vm.runInContext(layout.slice(layout.indexOf('  function setupChatDraft()'), layout.indexOf('  function positionRecorder()')), chatContext);
vm.runInContext('setupChatDraft()', chatContext);
field.value = 'draft A'; field.dispatchEvent();
field = new Field(); // New connection / DOM.
vm.runInContext('setupChatDraft()', chatContext);
assert.equal(field.value, 'draft A');
marker = {dataset:{session:'B', revision:''}};
field.value = ''; // React can reuse the same textarea between sessions.
vm.runInContext('setupChatDraft()', chatContext);
field.value = 'draft B'; field.dispatchEvent();
assert.equal(JSON.parse(storage.get('lass-chat-draft:A')).text, 'draft A');
assert.equal(JSON.parse(storage.get('lass-chat-draft:B')).text, 'draft B');
assert.equal(field.handlers.size, 1);
marker = {dataset:{session:'A', revision:'message-2'}};
field = new Field();
vm.runInContext('setupChatDraft()', chatContext);
assert.equal(field.value, ''); // Already submitted drafts must not come back.
assert.equal(storage.has('lass-chat-draft:A'), false);
for (const html of [editor, layout]) new vm.Script(html.match(/<script>([\s\S]*?)<\/script>/)[1]);
console.log('PASS: editor draft baselines, scope, cancellation, malformed storage; chat reload, session isolation and submitted-draft invalidation');
