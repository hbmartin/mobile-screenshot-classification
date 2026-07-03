// Download App Store screenshots for every app id in a CSV.
//
// Usage: node get_screenshots.js
//   INPUT_CSV   CSV whose first column is the app id (default: deduped.csv)
//   OUT_DIR     output directory (default: screenshots)
//   COUNTRY     storefront to scrape; different storefronts often carry
//               different screenshots (default: us)
//   CONCURRENCY parallel app downloads (default: 4)
//
// Progress is recorded in <OUT_DIR>/manifest.jsonl (one line per downloaded
// image with its URL and sha256), so an interrupted run can be re-run and
// will skip everything already fetched.
const store = require('app-store-scraper');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const Axios = require('axios');
const readline = require('readline');

const INPUT_CSV = process.env.INPUT_CSV || 'deduped.csv';
const OUT_DIR = process.env.OUT_DIR || 'screenshots';
const COUNTRY = process.env.COUNTRY || 'us';
const MAX_RETRIES = 4;
const MANIFEST = path.join(OUT_DIR, 'manifest.jsonl');

function parseConcurrency(value) {
    const parsed = Number.parseInt(value || '4', 10);
    return Number.isFinite(parsed) ? Math.max(1, parsed) : 4;
}

const CONCURRENCY = parseConcurrency(process.env.CONCURRENCY);

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function withRetries(fn, label) {
    for (let attempt = 0; ; attempt++) {
        try {
            return await fn();
        } catch (err) {
            if (attempt >= MAX_RETRIES) throw err;
            const delay = 1000 * 2 ** attempt;
            console.warn(`${label} failed (${err.message}); retrying in ${delay}ms`);
            await sleep(delay);
        }
    }
}

function completedKey(appId, url) {
    return `${appId}:${url}`;
}

function loadCompletedKeys() {
    const done = new Set();
    if (fs.existsSync(MANIFEST)) {
        for (const line of fs.readFileSync(MANIFEST, 'utf8').split('\n')) {
            if (!line.trim()) continue;
            try {
                const item = JSON.parse(line);
                if (item.appId && item.url) done.add(completedKey(item.appId, item.url));
            } catch {
                // Ignore a torn line from an interrupted run.
            }
        }
    }
    return done;
}

async function downloadImage(url, dir) {
    const response = await Axios({
        url,
        method: 'GET',
        responseType: 'arraybuffer',
        timeout: 30000,
    });
    fs.mkdirSync(dir, { recursive: true });
    const urlParts = url.split('/').slice(-2);
    const filepath = path.join(dir, urlParts[0].split('.')[0] + '_' + urlParts[1]);
    const data = Buffer.from(response.data);
    fs.writeFileSync(filepath, data);
    return {
        filepath,
        sha256: crypto.createHash('sha256').update(data).digest('hex'),
        bytes: data.length,
    };
}

async function fetchScreenshots(appId, done, manifestStream) {
    const app = await withRetries(() => store.app({ id: appId, country: COUNTRY }), `app ${appId}`);
    console.log(`${appId}: ${app.screenshots.length} screenshots`);
    const dir = path.join(OUT_DIR, String(appId));
    for (const url of app.screenshots) {
        const key = completedKey(appId, url);
        if (done.has(key)) continue;
        const info = await withRetries(() => downloadImage(url, dir), url);
        manifestStream.write(JSON.stringify({ appId, country: COUNTRY, url, ...info }) + '\n');
        done.add(key);
    }
}

async function main() {
    const ids = [];
    const rl = readline.createInterface({
        input: fs.createReadStream(INPUT_CSV),
        crlfDelay: Infinity,
    });
    for await (const line of rl) {
        const appId = line.split(',')[0].trim();
        if (appId) ids.push(appId);
    }

    fs.mkdirSync(OUT_DIR, { recursive: true });
    const done = loadCompletedKeys();
    const manifestStream = fs.createWriteStream(MANIFEST, { flags: 'a' });

    let next = 0;
    let failures = 0;
    async function worker() {
        while (next < ids.length) {
            const appId = ids[next++];
            try {
                await fetchScreenshots(appId, done, manifestStream);
            } catch (err) {
                failures++;
                console.error(`FAILED app ${appId}: ${err.message}`);
            }
        }
    }
    await Promise.all(Array.from({ length: CONCURRENCY }, worker));
    manifestStream.end();

    console.log(`Finished ${ids.length} apps; ${failures} failed.`);
    if (failures > 0) process.exitCode = 1;
}

main();
