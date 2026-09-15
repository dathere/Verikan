// Provenance and follow-up controls shared by fresh and saved chat answers.
// All helpers are pure so restored conversations render the same evidence.
(function (root) {
    'use strict';

    const MISSING = 'Not provided';
    const REVIEW_STATUSES = new Set(['reviewed', 'pending', 'unreviewed', 'unknown']);

    function text(value) {
        return typeof value === 'string' ? value.trim() : '';
    }

    function object(value) {
        return value && typeof value === 'object' && !Array.isArray(value) ? value : {};
    }

    function escape(value) {
        return String(value ?? '').replace(/[&<>"']/g, character => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
        }[character]));
    }

    function safeUrl(value) {
        const candidate = text(value);
        if (!/^https?:\/\//i.test(candidate)) return '';
        try {
            const url = new URL(candidate);
            if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) return '';
            return url.href;
        } catch (_) {
            return '';
        }
    }

    function normalizeSources(items) {
        const seen = new Set();
        return (Array.isArray(items) ? items : []).flatMap(item => {
            if (!item || typeof item !== 'object') return [];
            const url = safeUrl(item.url);
            const name = text(item.name) || text(item.dataset_title) || text(item.source)
                || (url ? new URL(url).hostname : '');
            if (!name) return [];
            const key = `${name.toLowerCase()}|${url}`;
            if (seen.has(key)) return [];
            seen.add(key);
            return [{ name, url, description: text(item.description) }];
        });
    }

    function normalizePeriod(value) {
        const period = object(value);
        const start = text(period.start);
        const end = text(period.end);
        return start || end ? { start: start || null, end: end || null } : null;
    }

    // verifiedMatch must be the answer actually served, never a related search
    // suggestion. A response's canonical evidence (even null dates) wins over
    // legacy metadata. In particular, chat creation/access dates are not data
    // retrieval dates, and a human review date is not a source update date.
    function normalizeEvidence(data, verifiedMatch) {
        const response = object(data);
        const match = object(verifiedMatch);
        const notebook = object(response.notebook || response.notebook_json);
        const metadata = object(object(notebook.metadata).data_concierge);
        const canonical = [response.evidence, metadata.evidence, match.evidence]
            .find(value => value && typeof value === 'object' && !Array.isArray(value));
        const legacy = { ...metadata, ...match, ...response };
        const evidence = canonical || legacy;
        const sources = canonical ? evidence.sources : [
            ...(Array.isArray(legacy.sources) ? legacy.sources : []),
            ...(Array.isArray(legacy.source_links) ? legacy.source_links : []),
            ...(Array.isArray(legacy.sourceLinks) ? legacy.sourceLinks : []),
            ...(Array.isArray(legacy.citations) ? legacy.citations : [])
        ];
        if (!canonical && !sources.length && text(legacy.data_source)) {
            sources.push({ name: legacy.data_source });
        }
        const status = text(evidence.verification_status);
        return {
            sources: normalizeSources(sources),
            data_period: normalizePeriod(evidence.data_period),
            retrieved_at: text(evidence.retrieved_at) || null,
            source_updated_at: text(evidence.source_updated_at) || null,
            verified_at: text(evidence.verified_at) || null,
            verification_status: REVIEW_STATUSES.has(status) ? status : 'unknown',
            original_query: text(evidence.original_query) || null
        };
    }

    function dateLabel(value) {
        const raw = text(value);
        // Limit parsing to unambiguous ISO dates; preserve year/month precision
        // rather than turning an unknown day into January 1 / the first day.
        if (/^\d{4}$/.test(raw)) return raw;
        if (/^\d{4}-(0[1-9]|1[0-2])$/.test(raw)) return raw;
        if (!/^\d{4}-\d{2}-\d{2}(?:T|$)/.test(raw)) return MISSING;
        const date = new Date(`${raw.slice(0, 10)}T00:00:00Z`);
        if (Number.isNaN(new Date(raw).getTime()) || Number.isNaN(date.getTime())
            || date.toISOString().slice(0, 10) !== raw.slice(0, 10)) return MISSING;
        const options = { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' };
        return date.toLocaleDateString('en-US', options);
    }

    function periodLabel(period) {
        if (!period) return MISSING;
        if (period.start && period.start === period.end) return period.start;
        if (period.start && period.end) return `${period.start} – ${period.end}`;
        return period.start ? `${period.start} · end not provided`
            : `Start not provided · ${period.end}`;
    }

    function renderEvidence(value) {
        const evidence = normalizeEvidence({ evidence: object(value) });
        const sources = evidence.sources;
        const sourceSummary = sources.length ? sources.slice(0, 2).map(item => item.name).join(', ')
            + (sources.length > 2 ? ` +${sources.length - 2}` : '') : MISSING;
        const period = periodLabel(evidence.data_period);
        const reviewLabels = {
            reviewed: 'Human reviewed', pending: 'Awaiting human review',
            unreviewed: 'Not human reviewed', unknown: MISSING
        };
        const reviewSummary = evidence.verification_status === 'reviewed'
            ? `Reviewed: ${dateLabel(evidence.verified_at)}`
            : evidence.verification_status === 'pending' ? 'Review pending' : '';
        const reviewNote = evidence.verification_status === 'reviewed'
            ? 'Human review checks the saved answer. It does not mean the underlying data is the latest available.'
            : 'Data period describes the observations. Retrieval, source updates, and human review are recorded separately.';
        const sourceList = sources.length ? `<ul class="answer-evidence-sources">${sources.map(source => `
            <li>${source.url ? `<a href="${escape(source.url)}" target="_blank" rel="noopener noreferrer">${escape(source.name)}<i class="bi bi-box-arrow-up-right" aria-hidden="true"></i></a>` : `<span>${escape(source.name)}</span>`}
            ${source.description ? `<span class="answer-evidence-description">${escape(source.description)}</span>` : ''}</li>
        `).join('')}</ul>` : `<p class="answer-evidence-missing">${MISSING}</p>`;
        const row = (label, content) => `<div><dt>${label}</dt><dd>${escape(content)}</dd></div>`;
        return `<details class="answer-evidence">
            <summary>
                <span class="answer-evidence-heading"><i class="bi bi-journal-check" aria-hidden="true"></i>Sources &amp; data dates</span>
                <span class="answer-evidence-summary"><span>Sources: ${escape(sourceSummary)}</span><span>Data period: ${escape(period)}</span>${reviewSummary ? `<span>${escape(reviewSummary)}</span>` : ''}</span>
            </summary>
            <div class="answer-evidence-body">
                <h3>Sources</h3>${sourceList}
                <dl class="answer-evidence-dates">
                    ${row('Data period', period)}
                    ${row('Data retrieved', dateLabel(evidence.retrieved_at))}
                    ${row('Source last updated', dateLabel(evidence.source_updated_at))}
                    ${row('Review status', reviewLabels[evidence.verification_status])}
                    ${row('Human reviewed on', dateLabel(evidence.verified_at))}
                </dl>
                ${evidence.original_query ? `<p class="answer-evidence-query"><strong>Original question</strong> ${escape(evidence.original_query)}</p>` : ''}
                <p class="answer-evidence-note">${reviewNote}</p>
            </div>
        </details>`;
    }

    function followupsFor(value, userQuery) {
        const message = object(value);
        const content = text(message.content) || text(message.answer);
        const status = text(message.status).toLowerCase();
        const confidence = text(message.confidenceLevel || message.confidence_level).toLowerCase();
        if (!content || message.error || message.isError || message.isStopped || message.stopped
            || message.clarification_needed || message.clarificationNeeded || message.isClarification
            || message.isRecommendation || message.isEscalation || message.requiresLogin
            || ['error', 'cancelled', 'canceled', 'stopped', 'failed', 'clarification'].includes(status)
            || ['error', 'stopped'].includes(confidence)) return [];
        // Older saved chats lack explicit failure flags. Do not frame a failed
        // query, a sign-in prompt, or a clarification as a successful result.
        if (/^(?:query (?:was )?(?:stopped|cancelled|canceled)|please (?:sign|log) in|i (?:could not|couldn't|wasn't able to|was unable to)|sorry[,!.])/i.test(content)) return [];
        const query = text(userQuery) || text(message.userQuery)
            || text(object(message.evidence).original_query) || text(message.verifiedQuery);
        if (!query) return [];
        // Keep each request self-contained when the server cannot resolve chat
        // references. These ask new questions, never edits to a missing notebook.
        let topic = query;
        // Our previous action may itself be the preceding user message. Unwrap
        // only our own complete prompt form so repeated clicks retain the base
        // topic instead of accumulating quoted follow-up instructions.
        while (true) {
            const previous = topic.match(/^Regarding my question “([\s\S]*)”: [\s\S]+$/);
            if (!previous) break;
            topic = previous[1];
        }
        const ask = question => {
            const prefix = 'Regarding my question “';
            const suffix = `”: ${question}`;
            // QueryRequest caps input at 2,000 characters. Count UTF-16 code
            // units conservatively and leave room for the action itself.
            const budget = 2000 - prefix.length - suffix.length;
            const context = topic.length > budget ? topic.slice(0, budget - 1).trimEnd() + '…' : topic;
            return prefix + context + suffix;
        };
        const explain = { label: 'Explain the result', query: ask('What does this result mean, and what limitations should I consider?') };
        if (/\b(methodology|definition|defined|measured|calculated|method|how (?:is|are|do|does))\b/i.test(query)) {
            return [
                { label: 'Show an example', query: ask('Can you walk through a worked example of this measurement or method?') },
                { label: 'Find source data', query: ask('Which original datasets and documentation support this method?') },
                { label: 'Explain limitations', query: ask('What assumptions and limitations should I consider when using this measure?') }
            ];
        }
        if (/\b(find data|datasets?|data sources|where can i find|available data)\b/i.test(query)) {
            return [
                { label: 'Explore coverage', query: ask('Which places, dates, and measures do these datasets cover?') },
                { label: 'Show sample records', query: ask('Can you show a small sample from the most relevant available dataset and explain its fields?') },
                { label: 'Check update schedule', query: ask('How often are these datasets updated, according to their source documentation?') }
            ];
        }
        const trend = { label: 'Show the trend', query: ask('How has the same measure changed over the available years, using comparable data?') };
        const periods = { label: 'Compare periods', query: ask('How does this result compare with the preceding available period, using the same measure and places?') };
        const geographical = /\b(neighbou?rhoods?|counties|county|states?|cities|city|boroughs?|districts?|in|across|within)\b/i.test(query);
        const groups = /\b(age|gender|race|ethnicity|demographic|groups?)\b/i.test(query);
        const compare = groups
            ? { label: 'Compare groups', query: ask('How does this measure differ across the demographic groups covered by the source, for the same period and places?') }
            : geographical
                ? { label: 'Compare places', query: ask('Which comparable places does the source cover, and how does this measure compare across them for the same period?') }
                : periods;
        if (/\b(trend|over time|historical|growth|changed|available years)\b/i.test(query)) {
            return [compare, { label: 'Latest available data', query: ask('What is the latest available value for the same measure and places, and which period does it describe?') }, explain];
        }
        if (/\b(compare|comparison|versus|vs\.?|difference between)\b/i.test(query)) {
            return [trend, periods, explain];
        }
        return [trend, compare, explain];
    }

    function renderFollowups(value) {
        const seen = new Set();
        const items = (Array.isArray(value) ? value : []).filter(item => {
            if (!item || !text(item.label) || !text(item.query)) return false;
            const key = item.query.trim().toLowerCase();
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
        }).slice(0, 3);
        if (!items.length) return '';
        return `<div class="answer-followups" role="group" aria-label="Explore this answer">
            ${items.map(item => `<button type="button" class="answer-followup" data-action="followup" data-question="${escape(item.query)}">${escape(item.label)}<i class="bi bi-arrow-up-right" aria-hidden="true"></i></button>`).join('')}
        </div>`;
    }

    const api = { normalizeEvidence, renderEvidence, followupsFor, renderFollowups };
    root.AnswerDetails = api;
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
