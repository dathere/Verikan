/* Read the query event stream without assuming network chunk boundaries. */
(function (root) {
    'use strict';

    async function consume(response, onProgress, signal) {
        if (!response.ok) {
            const error = new Error('The query request was not accepted.');
            error.status = response.status;
            throw error;
        }
        if (!response.body) throw new Error('The query stream is unavailable.');
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let result;
        let receivedResult = false;
        const cancel = () => { reader.cancel().catch(() => {}); };
        signal?.addEventListener('abort', cancel, { once: true });

        function receive(frame) {
            const lines = frame.split(/\r?\n/)
                .filter(line => line.startsWith('data:'))
                .map(line => line.slice(5).replace(/^ /, ''));
            if (!lines.length) return; // SSE heartbeat comments.
            const event = JSON.parse(lines.join('\n'));
            if (event.type === 'progress' && typeof event.message === 'string') {
                onProgress(event);
            } else if (event.type === 'result') {
                if (!event.data || typeof event.data.answer !== 'string') {
                    throw new Error('The query returned an incomplete answer.');
                }
                result = event.data;
                receivedResult = true;
            } else if (event.type === 'error') {
                const error = new Error('The analysis could not finish.');
                error.status = Number(event.status) || 500;
                throw error;
            }
        }

        try {
            signal?.throwIfAborted();
            while (!receivedResult) {
                const { value, done } = await reader.read();
                signal?.throwIfAborted();
                buffer += decoder.decode(value, { stream: !done });
                let boundary;
                while ((boundary = /\r?\n\r?\n/.exec(buffer))) {
                    const frame = buffer.slice(0, boundary.index);
                    buffer = buffer.slice(boundary.index + boundary[0].length);
                    receive(frame);
                    if (receivedResult) break;
                }
                if (done) {
                    if (!receivedResult && buffer.trim()) receive(buffer);
                    if (!receivedResult) throw new Error('The connection closed before an answer arrived.');
                    break;
                }
            }
            return result;
        } finally {
            signal?.removeEventListener('abort', cancel);
            await reader.cancel().catch(() => {});
            reader.releaseLock();
        }
    }

    const api = { consume };
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    else root.QueryStream = api;
})(globalThis);
