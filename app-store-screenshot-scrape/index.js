var store = require('app-store-scraper');

store.list({
    collection: store.collection.TOP_GROSSING_IOS,
    num: 200
})
    .then(apps => {
        apps.forEach(app => {
            console.log(`${app.id}, ${app.appId}, ${app.title}`);
            // store.app({ id: app.id }).then(console.log);
        });
    })
    .catch(console.log);
