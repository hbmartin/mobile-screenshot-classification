const store = require('app-store-scraper');
const fs = require('fs');
const Axios = require('axios')
const readline = require('readline');

async function downloadImage(url, dir) {
    console.log(`Fetch: ${url}`);
    const response = await Axios({
        url,
        method: 'GET',
        responseType: 'stream'
    });
    console.log(response.status)
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir);
    }
    let urls_parts = url.split("/").slice(-2);
    let filepath = dir + "/" + urls_parts[0].split(".")[0] + "_" + urls_parts[1];
    return new Promise((resolve, reject) => {
        response.data.pipe(fs.createWriteStream(filepath))
            .on('error', reject)
            .once('close', () => resolve(filepath));
    });
}

let ids = []

function fetchScreenshots(appId) {
    console.log(`Fetching screenshots for ${appId}`)
    store.app({ id: appId })
        .then(app => {
            console.log(`${app.screenshots.length} screenshots`);
            return Promise.all(
                app.screenshots.map(screenshot => {
                    downloadImage(screenshot, `${appId}`);
                })
            );
        })
        .then(results => {
            console.log(results);
            let nextId = ids.pop();
            if (nextId) {
                setTimeout(fetchScreenshots, 1000, nextId);
            }
        });
}

async function processLineByLine() {
    const fileStream = fs.createReadStream('deduped.csv');

    const rl = readline.createInterface({
        input: fileStream,
        crlfDelay: Infinity
    });

    for await (const line of rl) {
        let appId = line.split(",")[0]
        console.log(`Line from file: ${appId}`);
        ids.push(appId);
    }
    fetchScreenshots(ids.pop());
}

processLineByLine();
