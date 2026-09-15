const { test } = require('node:test');
const assert = require('node:assert/strict');
const { consume } = require('../../src/data_concierge/ui/static/js/query-stream.js');

function responseFromChunks(chunks) {
    return new Response(new ReadableStream({
        start(controller) {
            for (const chunk of chunks) controller.enqueue(chunk);
            controller.close();
        },
    }));
}

test('decodes split UTF-8 events, CRLF delimiters, heartbeat and result', async () => {
    const bytes = new TextEncoder().encode(': ping\r\n\r\ndata: {"type":"progress","message":"Checking Montréal"}\r\n\r\ndata: {"type":"result","data":{"answer":"Ready"}}\r\n\r\n');
    const chunks = Array.from(bytes, byte => new Uint8Array([byte]));
    const updates = [];
    assert.deepEqual(await consume(responseFromChunks(chunks), event => updates.push(event.message)), { answer: 'Ready' });
    assert.deepEqual(updates, ['Checking Montréal']);
});

test('rejects early disconnect instead of displaying a partial result', async () => {
    const bytes = new TextEncoder().encode('data: {"type":"progress","message":"Working"}\n\n');
    await assert.rejects(consume(responseFromChunks([bytes]), () => {}), /before an answer/);
});

test('abort cancels a reader waiting for the next server event', async () => {
    let cancelled = false;
    const response = new Response(new ReadableStream({ cancel() { cancelled = true; } }));
    const controller = new AbortController();
    const pending = consume(response, () => {}, controller.signal);
    controller.abort();
    await assert.rejects(pending, { name: 'AbortError' });
    assert.equal(cancelled, true);
});

test('server errors and malformed result payloads are not treated as answers', async () => {
    for (const event of [{ type: 'error', message: 'failed' }, { type: 'result', data: {} }]) {
        const bytes = new TextEncoder().encode(`data: ${JSON.stringify(event)}\n\n`);
        await assert.rejects(consume(responseFromChunks([bytes]), () => {}));
    }
});
