# Klasyfikator KNN od podstaw

Zaimplementuj od podstaw klasyfikator K-Nearest Neighbor na słynnym zbiorze danych iris-flower. Algorytm ten określa wartość niewidzianego punktu danych na podstawie wartości jego sąsiadów. W celu określenia odległości między nowym punktem a najbliższymi danymi obliczana jest odległość euklidesowa, będąca rozwinięciem twierdzenia Pitagorasa. W zależności od tego, jak wiele punktów znajduje się w pobliżu, odpowiednio klasyfikuje nowy punkt danych. KNN może obsługiwać klasyfikację binarną i więcej niż "n" liczb klas.

### Zalety i wady KNN

Najlepszym atrybutem klasyfikatora KNN jest jego względna prostota. Nie jest on trudny do zaimplementowania, a jego intuicja jest również bardzo prosta. Największą wadą KNN jest nieefektywność obliczeniowa. Algorytm iteruje nad każdym punktem danych, co sprawia, że jego implementacja na dużych zbiorach danych jest kłopotliwa. Problem ten można zmniejszyć, wektoryzując implementację, co jest łatwe do wykonania w MATLABie lub Octave, ale wymaga więcej zależności w Pythonie lub C/C++. Inną wadą jest to, że bardzo trudno jest reprezentować cechy wykraczające poza 2 lub 3 wymiary i ich granice decyzyjne, co sprawia, że KNN staje się swego rodzaju czarną skrzynką.

# Zależności:

* `train_test_split` z `sklearn.cross_validation`

* `accuracy_score` z `sklearn.metrics`.

* `numpy` do obliczania odległości między punktami

* `datasets` z `sklearn`

* `datasets.load_iris()`.

# Wyniki:

Testując szereg wartości k, model konsekwentnie uzyskuje wynik powyżej 92%, ale waha się wokół średniej 94%.  Najlepszy wynik wyniósł 97% przy 'k = 7', co oznacza, że nowy punkt danych jest porównywany z siedmioma najbliższymi sąsiadami. Można to jeszcze dopracować, ale jestem zadowolony z tych wyników. Dalszym usprawnieniem byłaby wizualizacja modelu, ale to może być trudne w przypadku danych wielowymiarowych.  

Przetłumaczono z www.DeepL.com/Translator (wersja darmowa)