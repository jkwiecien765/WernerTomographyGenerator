dataJK to folder do którego trafią pliki .csv z DataFrame zawierające interesujące dane
Wszystkie ważne funkcje są w paczce myPackage
W pliku werner_generator.ipynb jest jedna komórka z wywołaniem data_save_iterator. Parametr N to liczba stworzonych dataframe, n to liczba próbek w każdej z nich. Na obliczenia w pracowni proponuję je ustawić na N=2000, n=500. Po każdej z N iteracji plik z DataFrame jest zapisywany, więc przerwanie procesu nie spowoduje utraty dużej ilości danych.
Parametr 'Prefix' w data_save_iterator umożliwia dodanie prefixu do plików, co się przyda do odróżnienia plików z różnych wątków i maszyn.
Przy parametrach n=2, N=2 liczyło się u mnie prawie 3 minuty ale to podlega dużej fluktuacji ze względu na czas zbiegania metody optymalizacyjnej.