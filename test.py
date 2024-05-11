# from keras.models import load_model
from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences

# # Load the tokenizer and model
# tokenizer = Tokenizer()
# model = load_model('LSModel.h5')  # Replace with the correct path

# # Sample news articles for testing
# news_articles = [
#     "A recent study suggests that regular exercise and a healthy diet can significantly reduce the risk of chronic diseases such as heart disease and diabetes.",
#     "Scientists have discovered a new species of marine life deep in the ocean depths.",
#     "Breaking news: The moon is made of green cheese, according to a reliable source."
# ]

# # Assuming you have already defined the function for preprocessing text
# def preprocess_text(text):
#     # Perform any necessary text preprocessing here
#     # Tokenization, removing stop words, etc.
#     return text

# # Make predictions
# for article in news_articles:
#     preprocessed_article = preprocess_text(article)
#     sequences = tokenizer.texts_to_sequences([preprocessed_article])
#     padded_sequence = pad_sequences(sequences, maxlen=512)  # Adjust maxlen based on your model
#     prediction = model.predict(padded_sequence)
    
#     # Interpret the model's output directly
#     prediction_label = 'REAL' if prediction[0][0] >= 0.5 else 'FAKE'
    
#     print(f"Article: {article}")
#     print(f"Model Output: {prediction[0][0]}, Predicted Label: {prediction_label}")
#     print()





false_list = []

mx = ["""Rockstar has released the trailer of Grand Theft Auto VI, the next game in the blockbuster video game franchise a day earlier than expected. Unfortunately you'll have to wait until at least some point in 2025 to play it.
It's been a decade since Rockstar Games released Grand Theft Auto V. While fans have been more than able to keep themselves busy with GTA Online and a few re-releases, they've been waiting patiently (or impatiently) for more single-player action. The wait now has a theoretical end with Rockstar revealing the first official look at the game and a 2025 release window.
As indicated by a recent teaser image, leaks and various rumors, GTA VI will be set in Leonida, Rockstar's take on Florida, and largely centered on Vice City, the series' stand in for Miami. Given that the trailer features a ton of Instagram-style live streaming, GTA VI seems to be a contemporary game, rather than one set in the '80s like 2002's Grand Theft Auto: Vice City. It remains to be seen if those streams are an integral game mechanic, purely narrative tool or just an aesthetic choice for the trailer, though.
It also appears that the game will have a playable female character, Lucia, for the first time in the modern incarnation of the franchise, just as the rumors predicted. Other highlights of the trailer include Florida's swampy Everglades National Park, an airboat, some wildlife and, of course, a strip club.
There's almost zero detail about the broader story of GTA VI, other than Lucia being in prison, presumably at the start of the game. But there are plenty of glimpses of the kind of shenanigans you'll be able to get up to, including your usual robberies and car chases. There's also brief shot of an alligator wandering into gas station store — gut instinct says your character might be behind that. Unfortunately it'll be over a year before we know for sure.
Mint Mobile emailed customers this weekend alerting them that their information may have been stolen during a security breach. BleepingComputer reports that names, phone numbers, email addresses, plan descriptions, and SIM and IMEI numbers were accessed.
On December 11, NASA successfully beamed an ultra-high definition video from the Psyche spacecraft to Earth. At the time, Psyche was about 19 million miles away. The video signal was received 101 seconds after it was sent.
VX-Underground shared this week that hackers were able to breach Ubisoft's internal services in an attempt to exfiltrate 900GB of data. Ubisoft shut down the breach after 48 hours, and told BleepingComputer it's investigating the incident.
For the final installment of Hitting the Books for 2023, we're bringing you an excerpt from the fantastic Material World: The Six Raw Materials That Shape Modern Civilization by Ed Conway.
GM has paused deliveries of the new Chevy Blazer EV after drivers found the infotainment system keeps crashing and displaying all kinds of error messages. The company said it is aware of "software quality issues" and is working on a fix.
Bluesky announced this week that you can now view posts on from the social network without logging in. It's also overhauled its logo, replacing the cloudy blue sky with a simple blue butterfly.
This week: The Apple Watch ban is here,  Samsung adds foldables to its self-repair program for the first time,  Sony has sold 50 million PS5 consoles over three years.
Apple has reportedly started negotiating with major publishers and news organizations to ask for permission to use their content to train the generative AI system it's developing.
Here's a list of the best Nintendo Switch controllers you can buy right now, as chosen by Engadget editors.
The Humane AI Pin is expected to start shipping in March. The company posted on Friday that “those who placed priority orders will receive their Ai Pins first when we begin shipping in March.”
Insomniac Games has weighed in publicly for the first time since hackers leaked over 1.3 million of the publisher’s private files. The studio posted that it’s “saddened and angered” by the cyberattack, describing the internal aftermath as “extremely distressing.”
This week's best tech deals include the Apple MacBook Air M2 for $1,299, the Apple AirTag for $24, the LG A2 OLED TV for $550 and a ton of discounts on good video games.
Here's a list of the best handheld gaming systems you can buy, as chosen by Engadget editors.
We explain why Apple had to stop selling the Apple Watch Series 9 and Ultra 2 this Christmas.
First American, a real estate and mortgage financial firm, experienced a "cybersecurity incident" impacting operations, the company posted on its website on Thursday.
Intuit is shutting down its popular Mint app in 2024. Engadget tested a bunch of popular alternatives. Here are our favorites.
Here's a list of the best iPhone accessories you can buy, as chosen by Engadget editors.
The biggest news stories this morning:  Hyperloop One is shutting down, Microsoft is nixing its Windows mixed-reality platform, Netflix milks Squid Game again with a $39 in-person ‘experience’.
Tesla has issued a second recall in the US in as many weeks. The company has issued an over-the-air update to resolve a issue that could increase the risk of injury during a crash.
The third season of Formula E's Unplugged docuseries is coming to Roku in January. The motorsport's new streaming home will start airing live races later that month.
Subscribe to our two newsletters:
 - A weekly roundup of our favorite tech deals
 - A daily dose of the news you need
Please enter a valid email address
Please select a newsletter
By subscribing, you are agreeing to Engadget's Terms and Privacy Policy."""]
# Assuming your new text is stored in the variable 'new_text'
# for new_text in fake_news.tolist():
import numpy as np
from keras.preprocessing.sequence import pad_sequences

# Assuming your model is named model4
for new_text in mx:
    # Preprocess the new text
    tokenized_new_text = tokenizer.texts_to_sequences([new_text])
    padded_new_text = pad_sequences(tokenized_new_text, maxlen=512)  # Ensure consistent maxlen value

    # Use the model to predict the probability
    predicted_prob = model4.predict(padded_new_text)

    # Adjust the threshold for classification
    threshold = 0.5
    predicted_label = 'REAL' if predicted_prob[0][0] >= threshold else 'FAKE'

    print(f"Predicted Label: {predicted_label}")
