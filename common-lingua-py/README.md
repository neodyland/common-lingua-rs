# common-lingua

Simple language detect module
```py
from common_lingua import detect

for lang, prob in detect("おはようございます。").items():
    if prob > 0.01:
        print(lang, prob)
# jpn 0.998576283454895
```
