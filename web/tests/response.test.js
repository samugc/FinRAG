import test from 'node:test';
import assert from 'node:assert/strict';
import { statusLabel, citationParts } from '../src/response.js';

for (const status of ['produced','model_abstention','invalid_output','no_evidence','backend_error','insufficient_context']) {
  test(`explicit state ${status} overrides compatibility boolean`, () => {
    assert.equal(statusLabel({status,sufficient_evidence:true}), statusLabel({status,sufficient_evidence:false}));
    assert.ok(statusLabel({status}));
  });
}
test('grouped references preserve surrounding response text and indices', () => {
  assert.deepEqual(citationParts('Value [1, 2].'), [
    {text:'Value ',ids:[]}, {text:'[1, 2]',ids:[1,2]}, {text:'.',ids:[]},
  ]);
});
test('unsupported citation syntax remains visible as raw text', () => {
  assert.deepEqual(citationParts('[1-3]'), [{text:'[1-3]',ids:[]}]);
});
test('empty response is renderable', () => {
  assert.deepEqual(citationParts(), [{text:'',ids:[]}]);
});
