from common_lingua import Model

model = Model()

for lang, prob in model.detect("おはようございます").items():
    if prob > 0.01:
        print(lang, prob)
