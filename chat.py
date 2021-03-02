import random
import json
import time
from selenium import webdriver
from simon.accounts.pages import LoginPage
from simon.chat.elements import MessageWriter
from simon.chats.pages import PanePage
from simon.header.pages import HeaderPage
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

driver = webdriver.Chrome()
driver.maximize_window()
login_page = LoginPage(driver)
login_page.load()
time.sleep(4)
pane_page = PanePage(driver)
me_contact = pane_page.get_me_contact()
open_chat = pane_page.opened_chats
contact_list = ["Dnonit"]
message_writer = MessageWriter(driver)
first_time_contact = []
contact_replay = None

while True:
    while me_contact.last_message != "stop" and contact_replay is None:
        for current_chat in open_chat:
            if current_chat.name in contact_list and current_chat.has_notifications():
                contact_replay = current_chat
                break
    if me_contact.last_message == "stop":
        break
    contact_replay.click()
    if contact_replay.name in first_time_contact:
        sentence = tokenize(contact_replay.last_message)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    message_writer.send_msg(random.choice(intent['responses']))
        else:
            message_writer.send_msg("I do not understand...")
    else:
        message_writer.send_msg(f"Hello {contact_replay.name}, Tamir is"
                                f" busy right now, I'm his helping"
                                f"bot can I help you ?")
        first_time_contact.append(contact_replay.name)
    me_contact.click()
    contact_replay = None

header_page = HeaderPage(driver)
header_page.logout()
driver.quit()
