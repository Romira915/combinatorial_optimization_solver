use serde_json::Value;
use serenity::{
    http::client::Http,
    model::{channel::Embed, webhook::Webhook as SerenWebhook},
    prelude::HttpError,
    Error,
};

pub struct Webhook {
    token: String,
    id: u64,
}

impl Webhook {
    pub fn new(url: &str) -> Self {
        let mut rsplit = url.rsplit("/");
        let token = rsplit.next().unwrap().to_string();
        let id = rsplit.next().unwrap().parse::<u64>().unwrap();

        Webhook { token, id }
    }

    pub async fn send(&self, embed: Value) -> Result<(), Error> {
        let http = Http::default();
        let webhook = http.get_webhook_with_token(self.id, &self.token).await?;

        let _ = webhook
            .execute(&http, false, |w| w.embeds(vec![embed]))
            .await?;

        Ok(())
    }
}
