// Run with: node --test tests/ui/answer-details.test.cjs
const test = require('node:test');
const assert = require('node:assert/strict');
const { normalizeEvidence, renderEvidence, followupsFor, renderFollowups } = require('../../src/data_concierge/ui/static/js/answer-details.js');

test('cache access and query dates cannot be mistaken for retrieval or observation dates', () => {
    const evidence = normalizeEvidence({
        answer: 'An answer about 2024', query: 'Latest values for 2025?',
        timestamp: '2026-09-15T13:00:00Z', created_at: '2026-09-15',
        notebook: { metadata: { data_concierge: { created_at: '2026-09-15' } } }
    });
    assert.equal(evidence.retrieved_at, null);
    assert.equal(evidence.source_updated_at, null);
    assert.equal(evidence.data_period, null);
    assert.equal(evidence.verified_at, null);
    assert.equal(evidence.verification_status, 'unknown');
    assert.match(renderEvidence(evidence), /Data period: Not provided/);
});

test('canonical evidence preserves known review and unknown source dates on saved answers', () => {
    const evidence = normalizeEvidence({
        retrieved_at: '2026-09-15T13:00:00Z',
        evidence: {
            sources: [{ name: 'Census', url: 'https://www.census.gov/' }],
            data_period: { start: '2020', end: '2024' },
            retrieved_at: null, source_updated_at: null,
            verified_at: '2025-03-01T12:00:00Z', verification_status: 'reviewed'
        }
    }, { retrieved_at: '2026-09-15', data_period: { start: '2026', end: '2026' } });
    assert.equal(evidence.retrieved_at, null);
    assert.equal(evidence.data_period.start, '2020');
    const html = renderEvidence(evidence);
    assert.match(html, /Reviewed: Mar 1, 2025/);
    assert.match(html, /Data period: 2020 – 2024/);
    assert.match(html, /<dt>Data retrieved<\/dt><dd>Not provided<\/dd>/);
    assert.match(html, /does not mean the underlying data is the latest/);
    assert.doesNotMatch(html, /2026/);
});

test('restored notebook evidence and an actually served verified record are supported', () => {
    const evidence = { sources: [], retrieved_at: '2025-01-12', verification_status: 'unreviewed' };
    assert.equal(normalizeEvidence({ notebook: { metadata: { data_concierge: { evidence } } } }).retrieved_at, '2025-01-12');
    assert.equal(normalizeEvidence({}, { evidence }).verification_status, 'unreviewed');
});

test('legacy source links are merged and deduplicated without inferring dates', () => {
    const source = { name: 'BLS', url: 'https://www.bls.gov/' };
    const evidence = normalizeEvidence({
        source_links: [source], sourceLinks: [source],
        citations: [{ dataset_title: 'Census survey', url: 'https://www.census.gov/', access_date: '2025-01-01' }]
    });
    assert.equal(evidence.sources.length, 2);
    assert.equal(evidence.retrieved_at, null);
});

test('untrusted source names, URLs, descriptions, periods, and queries cannot inject markup', () => {
    const evidence = normalizeEvidence({ evidence: {
        sources: [
            { name: '<img src=x onerror=alert(1)>', url: 'javascript:alert(1)', description: '</li><script>alert(1)</script>' },
            { name: 'Unsafe data URL', url: 'data:text/html,<script>alert(1)</script>' },
            { name: 'Protocol relative', url: '//evil.example/' },
            { name: 'Credentials', url: 'https://user:password@example.org/' },
            { name: 'Real source', url: 'https://data.example.org/?q=" onclick="alert(1)' }
        ],
        data_period: { start: '<img src=x>', end: '2024' },
        original_query: '<svg onload=alert(1)>', verification_status: 'reviewed'
    } });
    assert.equal(evidence.sources.filter(source => source.url).length, 1);
    const html = renderEvidence(evidence);
    assert.doesNotMatch(html, /<(?:img|script|svg)\b/);
    assert.doesNotMatch(html, /href="(?:javascript:|data:|\/\/)/);
    assert.doesNotMatch(html, /\sonclick="/);
    assert.match(html, /&lt;img src=x onerror=alert\(1\)&gt;/);
    assert.match(html, /rel="noopener noreferrer"/);
});

test('invalid dates stay missing and partial observation ranges keep their uncertainty', () => {
    const html = renderEvidence({
        data_period: { start: null, end: '2024' },
        retrieved_at: '2026-02-31', source_updated_at: 'not a date',
        verified_at: '2025-03', verification_status: 'pending'
    });
    assert.match(html, /Start not provided · 2024/);
    assert.match(html, /<dt>Data retrieved<\/dt><dd>Not provided/);
    assert.match(html, /<dt>Source last updated<\/dt><dd>Not provided/);
    assert.match(html, /<dt>Human reviewed on<\/dt><dd>2025-03/);
    assert.match(html, /Awaiting human review/);
    assert.doesNotMatch(html, /Mar 3, 2026/);
});

test('unsuccessful and clarification responses have no success followups', () => {
    const flags = [
        { error: 'failed' }, { isError: true }, { isStopped: true }, { stopped: true },
        { clarification_needed: true }, { clarificationNeeded: true }, { isClarification: true },
        { confidenceLevel: 'error' }, { confidence_level: 'error' },
        { status: 'cancelled' }, { status: 'failed' }, { isEscalation: true }, { requiresLogin: true }
    ];
    for (const flag of flags) {
        assert.deepEqual(followupsFor({ content: 'A response', ...flag }, 'Population in Texas?'), []);
    }
    assert.deepEqual(followupsFor({ content: "I wasn't able to fetch live data." }, 'Population?'), []);
    assert.deepEqual(followupsFor({ content: 'Query stopped.' }, 'Population?'), []);
    assert.deepEqual(followupsFor({ content: 'An answer' }, ''), []);
});

test('factual answer actions preserve the original topic and ask questions without notebook edits', () => {
    const query = 'What is the unemployment rate in Texas?';
    const followups = followupsFor({ content: 'The answer', isQuickAnswer: true, hadNotebook: false }, query);
    assert.deepEqual(followups.map(item => item.label), ['Show the trend', 'Compare places', 'Explain the result']);
    for (const item of followups) {
        assert.ok(item.query.includes(query));
        assert.doesNotMatch(item.query, /\b(edit|notebook|chart|California|2026)\b/);
    }
});

test('followups change with the question type instead of repeating an existing trend or comparison', () => {
    const answer = { content: 'The answer' };
    assert.deepEqual(followupsFor(answer, 'Show the population trend in Texas').map(item => item.label),
        ['Compare places', 'Latest available data', 'Explain the result']);
    assert.deepEqual(followupsFor(answer, 'Compare population in Texas and Ohio').map(item => item.label),
        ['Show the trend', 'Compare periods', 'Explain the result']);
    assert.equal(followupsFor(answer, 'How is unemployment measured?')[0].label, 'Show an example');
    assert.equal(followupsFor(answer, 'Find datasets about public transport')[0].label, 'Explore coverage');
    assert.equal(followupsFor(answer, 'What is employment by age group?')[1].label, 'Compare groups');
});

test('long questions and repeated clicks stay within the API limit without nesting context', () => {
    const answer = { content: 'The answer' };
    const longQuery = 'What is population in Texas? ' + 'Additional context. '.repeat(200);
    const actions = followupsFor(answer, longQuery);
    for (const action of actions) {
        assert.ok(action.query.length <= 2000);
        assert.ok(action.query.includes('What is population in Texas?'));
        assert.match(action.query, /…/);
    }
    let query = 'What is population in Texas?';
    for (let i = 0; i < 50; i += 1) {
        query = followupsFor(answer, query)[0].query;
        assert.ok(query.length <= 2000);
        assert.equal((query.match(/Regarding my question/g) || []).length, 1);
        assert.ok(query.includes('What is population in Texas?'));
    }
});

test('followup attributes are escaped and actions are delegated with a maximum of three unique questions', () => {
    const html = renderFollowups([
        { label: '<script>bad</script>', query: 'What about "Ohio"?\' onclick=\'alert(1)' },
        { label: 'Duplicate', query: 'What about "Ohio"?\' onclick=\'alert(1)' },
        { label: 'Second', query: 'Second?' },
        { label: 'Third', query: 'Third?' },
        { label: 'Fourth', query: 'Fourth?' }
    ]);
    assert.equal((html.match(/data-action="followup"/g) || []).length, 3);
    assert.doesNotMatch(html, /<script>|\sonclick=['"]/);
    assert.match(html, /&quot;Ohio&quot;/);
    assert.match(html, /&#39; onclick=&#39;/);
    assert.equal(renderFollowups([]), '');
});
