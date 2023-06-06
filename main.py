import customtkinter
import next_word_prediction
from tensorflow import keras

train_model = False
model_path = "model"

train_X, train_y, number_of_available_words, tokenizer = next_word_prediction.load_train_data("hp.txt")

if train_model:
    model = next_word_prediction.build_and_compile_the_model(train_X, train_y, number_of_available_words)
else:
    model = keras.models.load_model(model_path)

print("Model ready!")

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

root = customtkinter.CTk()
root.geometry("600x300")


def generate_prediction_words():
    predicted_words = next_word_prediction.predict_next_word(tokenizer, model, entry1.get(), train_X.shape[1])
    print(predicted_words)
    result_label.configure(text="suggested next words: " + ', '.join(str(x) for x in predicted_words))


frame = customtkinter.CTkFrame(master=root)
frame.pack(pady=20, padx=60, fill="both", expand=True)

label = customtkinter.CTkLabel(master=frame, text="Next word prediction", font=("TkDefaultFont", 20))
label.pack(pady=12, padx=10)

entry1 = customtkinter.CTkEntry(master=frame, placeholder_text="Start writing", width=350, height=60)
entry1.pack(pady=12, padx=10)

button = customtkinter.CTkButton(master=frame, text="Generate prediction", command=generate_prediction_words)
button.pack(pady=10, padx=10)

result_label = customtkinter.CTkLabel(master=frame, text="Suggested next words: ")
result_label.pack(pady=5, padx=10)

root.mainloop()
