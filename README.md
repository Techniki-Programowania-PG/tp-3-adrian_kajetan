# Projekt nr 3 – Przetwarzanie sygnałów z pybind11

## Grupa Adrian Protasiuk i Kajetan Rajczyk, ACiR 3B, grupa nr 7
## Indeksy kolejno: 203374 i 203627

# Ten projekt to biblioteka C++ do przetwarzania sygnałów, udostępniona jako moduł Pythona za pomocą pybind11.

## Funkcjonalności podstawowe:
- Wizualizacja sygnału z wykorzystaniem biblioteki matplotplusplus 
- DFT i transformata odwrotna 
- Filtracja 1D i 2D 
- Generowanie sygnałów o zadanej częstotliwości (sin, cos, prostokątny, piłokształtny)

## Funckjonalność dodatkowa (grupa nr 7):
- Usuwanie wysokich częstotliwości z sygnału (korzystajac z DFT)

## Instalacja (np. w git bash przy pomocy condy po sklonowaniu projektu i przejściu do jego lokalizacji):
1. conda create -n (nazwa_środowiska) python=(wybrana_wersja_pythona - np. 3.11) pip
2. conda activate (nazwa_środowiska)
3. pip install scikit-build-core pybind11 pytest cmake ninja build
4. pip install . 
5. instalacja biblioteki matplotplusplus na komputerze, gdzie folder matplotplusplus/build musi znaleźć się w lokalizacji projektu

# Wyjaśnienie kroków:
1. Stworzenie środowiska pythonowego 
2. Aktywowanie tego środowiska
3. Zainstalowanie w środowisku odpowiednich bibliotek potrzebnych do zbudowania modułu pythonowego
4. Zbudowanie projektu
5. Zainstalowanie biblioteki matplotplusplus jest konieczne, gdyż repozytorium uwzględnia je jedynie jako submoduł z oficjalnego repozytorium matplotplusplus

## Uruchomienie wcześniej przygotowanych testów po instalacji i budowie projektu:
- pytest -v -s tests

## Struktura Projektu:
- Pliki potrzebne do zbudowania modułu to przede wszystkim CmakeLists.txt, pyproject.toml i src/bindings.cpp
- W folderze /src znajdują się wszystki pliki .cpp oraz .hpp zawierające cały kod c++
- W głównym katalogu jest także folder raport, w którym znajduje się plik oddanie_projektu.ipynb z dokładnym opisem wszystkich funkcjonalności
- Plik bindings.cpp zawiera zbiór wszystkich możliwych funkcji wraz z ich krótkim opisem a także eksportuje on je do modułu pythonowego
- Widoczne są także foldery pybind11 i matplotplusplus będące bibliotekami niezbędnymi do kolejno: stworzenia pythonowego modułu oraz wizualizacji sygnałów
- Oprócz nich istnieje folder tests z jednym tylko plikiem test_basic.py zawierającym proste testy związane z funkcjonalnością projektu
