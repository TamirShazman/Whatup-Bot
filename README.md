# Whatup-Bot
This application is a Basic NLP project that use simple NN for learning.
The project consist the following files :
1. intents.json -> The so called "data-set", it's contains a list of list (dictionary ), that every list is a feeling 
   that the bot will learn. You can change the content of the file to teach the bot new feeling or new response.
2. nltk_utils.py -> A couple of NLP function (it's has comment inside).   
3. model.py -> The NN structure.
4. train.py -> Script for training the NN and modify the weight and bias of the NN.
5. chat.py -> open a web-drive using chromedriver.exe and handling the new messages.

The basic idea behind the bot is a NN that learn feeling through sentences from intents.json and adjust the NN. After 
running the train.py, you can run chat.py. it will open a Whatups chrome window with a barcode that you will need to open
with your phone (like you training to login to whatups via web). After that, the bot will start to work.
you can modify to which contact the bot will response to (in chat.py in contact list).

I made this project for better understanding of NLP and I publish it for comments, advise that will improve the Bot.

Thank you for your time, hope you will like it.

P.S. if you have any suggestions for improvement your more then welcome for leaving me a message. 