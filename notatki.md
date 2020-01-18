# Wykład 2
## Jak działają sieci neuronowe
*Feedforward neural network* (jednokierunkowa sieć neuronowa) - sieć neuronowa, w której nie ma cyklów (w przeciwieństwie do np. sieci rekurencyjnej).

We wszystkich sieciach można wyróżnić następujące warstwy:
- Input Layer
- Hidden Layers
- Output Layer  
Warstwy mogą posiadać tzw. `Bias node`, czyli węzeł, który zawsze ma wartość `1`. Warstwa bez takiego węzła nie będzie w stanie wyprodukować `niezerowych` (dla skali linearnej, mogą być to inne wartości zależne od funkcji aktywacji) danych dla zerowego wejścia. Dodanie takich węzłów wydłuży czas trenowania, zwiększy prawdopodobieństwo przeuczenia, ale czyni sieci bardziej elastyczne.

Przykłady warstw:
- Dense Layer - warstwa, w której neurony łączą się ze wszystkimi neuronami z poprzedniej warstwy
- Dropout Layer - warstwa, w której neurony są losowo wyłączane w trakcie uczenia. W trakcie testowania warstwa jest dezaktywowana. Zapobiega to przeuczeniu sieci.

Każda warstwa ma neurony (perceptrony), które dla danego wejścia produkują wyjście. Połączenia pomiędzy neuronami mają określone wagi. Wyjście neurona można opisać równaniem:
```
output = activation(weighted sum of inputs)
```

Przykłady funkcji aktywacji:
- Sigmoid - zmienia największe ujemne liczby na bliskie zeru, a największe dodatnie na bliskie jeden
- ReLU - zmienia wejscie na max(0, x)
- SoftMax - zmienia wejścia na prawdopodobieństwa, które sumują się do 1, często wykorzystywane w ostatnich warstwach sieci konwolucyjnych

## Jak trenowana jest sieć
Na samym początku procesu uczenia losowane są wagi. Uczenie polega na korekcji tych wag co każdą epokę.  
W trakcie każdej epoki pewien batch danych stanowiących wejście jest propagowane przez sieć. Na podstawie błędu (różnicy pomiędzy wyjściem sieci, a oczekiwanym wyjściem sieci) obliczane są gradienty poszczególnych wag. Pod koniec epoki są one sumowane i odejmowane dla każdej z wag.  

*Błąd* albo koszta można obliczyć za pomocą funkcji straty. Taką funkcją może być np. `MSE`, który oblicza średni błąd kwadratowy.
W celu obliczenia jak bardzo należy zmienić wagi wykorzystuje się gradienty funkcji straty dla poszczególnych wag. Te gradienty zależą od gradientów z warstw następujących, dlatego oblicza się je od końca sieci. Do tego wykorzystuje się algorytm *Backpropagatian*.  
Do obliczania wag wykorzystuje się często `Stochastic Gradient Descent` - każda epoka operuje tylko na podzbiorze danych trenujących. Tracimy wtedy nieznacznie na dokładności i potrzebujemy większej liczby epok na rzecz wykonania dużo mniejszej liczby obliczeń.

# Wykład 3
`One hot encoding` - proces, w którym kategorie (np. pies, kot, koza) są zamienane na ciąg binarny `one-hot` (posiadający jedną jedynke w zapisie).  
`Tensor` - macierz macierzy.  
`CNN` (Convolutional Neural Network - głęboka sieć konwolucyjna) - pozwala na filtrowanie różnych części danych uczących i wyostrzać ważne cechy, pozwala na rozpoznawanie i klasyfikacje pewnych wzorów. Wykorzystywane są głównie przy przetwarzaniu grafiki, wideo albo języka naturalnego.

W przypadku konwolucji 2D: rozmiar filtra zależy od rozmiaru konwolucji oraz liczby kanałów (dla obrazu RGB i konwolucji 2x2 będzie to 2x2x3).
Rozmiar wyjścia zależy od wielu parametrów takich jak typ konwolucji, liczba filtrów, padding, stride itp.
Liczba operacji to rozmiar wyjścia * rozmiar filtra.

W pierwszych warstwach konwolucje wykrywają proste elementy, np. krawędzie. W dalszych warstwach konwolucje wykrywają bardziej zaawansowane wzory.

`Flattening` - usuwanie wszystkich wymiarów z tensora oprócz jednego. Powiedzmy, że na wyjściu warstwy konwolucyjnej tensor o kształcie (24, 24, 48) i chcemy przekazać dane do FC Layer (Dense Layer) - musimy wpierw ten tensor spłaszczyć do kształtu (24*24*48).  
`Pooling` - redukcja rozmiaru np. poprzez uśrednianie pewnych obszarów. `Pooling` pozwala na:
- zmniejszenie liczby parametrów, a co za tym idzie zwiększa wydajność sieci,
- zapobieganie przeuczaniu sieci (ponieważ część informacji jest tracona),
- ekstrakcje cech z sąsiedztwa.

**Dzięki `poolingowi` model jest mniej wrażliwy na translacje.**  

- `Avg pooling` - uśrednia, rozmywa informacje, jest mniej stratny niż `Max pooling`.
- `Max pooling` - wybiera największe wartości, wyodrębnia najbardziej charakterystyczne cechy z sąsiedztwa, w bardziej ekstremalnych przypadkach może dać lepsze wyniki niż `Avg pooling`. Jest częściej wykorzystywany od `Avg pooling`.

## Parametry konwolucji i ich wpływ na działanie sieci
- `filters` - liczba filtrów, pozwala na wyodrębnienie większej liczby informacji (cech).
- `kernel size` - rozmiar filtra, małe filtry są bardziej szczegółowe, duże filtry pozwalają na wyodrębnienie bardziej globalnych cech. Jeśli różnice pomiędzy analizowanymi obiektami są małe, z reguły korzystamy z małych filtrów.
- `padding` - w przypadku `same` dodawane są dodatkowe kolumny, dzięki czemu wyjście warstwy konwolucyjnej jest takie samo jak jej wejście, w przeciwieństwie do `valid` gdy nie dodawane są żadne kolumny.
- `stride` - co ile pikseli ma się poruszać okno, duży `stride` znacznie zmniejsza liczbę parametrów.
- `dilation rate` - większe wartości niż 1 pozwalają na analizowanie danych w dalszym sąsiedztwie.

# Wykład 4
- `AlexNet` (2012):
  - boom na głębokie uczenie!
  - wykorzystuje `Dropout` jako walkę ze zjawiskiem przeuczenia
  - konwolucje 11x11
- `ZFNet` (2013):
  - *ulepszony AlexNet*
  - w porównaniu do `AlexNetu` wykorzystuje mniejsze konwolucje 7x7
- **`VGGNet` (2014):**:
  - w porównaniu do `AlexNetu` jest olbrzymi
  - zauważono, że zamiast jednej warstwy konwolucyjnej z dużym kernelem można użyć paru warstw z kernelem 3x3, co zwiększa wydajność sieci i lepiej zapobiega przeuczeniu
  - istnieje wiele wariacji z różną liczbą warstw
  - wariacja VGG-16 wykorzystuje konwolucje 1x1
- **`Inception` (2014):**:
  - wykorzystuje specjalne moduły `Inception`, w którym na tym samym poziome przeprowadzane są równolegle konwolucje 1x1, 3x3 i 5x5 oraz `max pooling`
    - na każdym poziomie wyodrębnione cechy są konkatenowane,
    - konwolucje i pooling mogą wykrywać globalne lub lokalne cechy - w trakcie trenowania sieci wykrywane jest, na jakich cechach nam zależy i dla połączeń do odpowiadających im konwolucji nadawane są większe wagi
  - konwolucje 1x1 w celu zmnieszenia rozmiaru modelu i redukcji problemu przeuczania
  - **`global average pooling`** - każdy kanał jest uśredniany do jednej wartości, dzięki czemu jest mniej podatny na przeuczenie
  - **`gradient vanishing problem`** - w celu aktualizacji wag wykorzystuje się gradient funkcji straty w zależności od poszczególnych wag. Może dojść do sytuacji, że ten gradient zanika, przez co sieć przestaje się uczyć. Bardziej narażone są na to sieci z większą liczbą warstw. Aby temu zapobiec w `Inception` jest parę *ostatnich warstw* z `SoftMax`, aby strata była kombinacją ich wszystkich, a nie tylko "ostatniej ostatniej" warstwy.
- `InceptionV3` (2015):
  - w miejsce konwolucji 5x5 używa dwóch konwolucji 3x3
  - ...a w miejsce konwolucji 3x3 używa dwóch konwolucji 3x1 i 1x3
  - zmniejszenie liczby parametrów w porównaniu do poprzednika
- `SqueezeNet` (2016):
  - celem było zmniejszenie `AlexNet`, co się udało (z 240MB parametrów do 5MB)
  - wykorzystują bloki / moduły `fire`
- **`ResNet` (2015)**:
  - początki bardzo głębokich sieci
  - używa konwolucji 1x1
  - brak warstw `pooling`
  - głębokość sieci wymagała znalezienia rozwiązania problemu z zanikającym gradientem - wykorzystuje się do tego `skip connection`:
    - wejścia niektórych warstw propagowane są na wyjścia innych z pominięciem paru warstw pomiędzy nimi
    - te połączenia pozwalają na szybsze przekazywanie gradientu, rozwiązując tym problem jego zanikania
  - `batch normalization` - normalizacja wszystkich cech danej warstwy do pewnego zakresu
    - aktywacje nie generują skrajnie małych i dużych wartości, co pozwala na użycie wyższego `learning rate`
    - dodaje trochę szumu bez utratu dużej ilości informacji, dzięki czemu zapobiega przeuczeniu i pozwala na zmniejszenie liczby warstw `dropout`
- `ResNeXt` (2016):
  - 32 razy wykorzystali bootleneck (?)
- `DenseNet` (2017):
  - propagacja wejścia na wszystkie wyjścia w danym bloku (bardziej skomplikowane bloki rezydualne)
  - silny "Gradient flow"
  - wydajny obliczeniowo
  - bardziej zróżnicowane cechy z warstw konwolucji
- **`MobileNet` (2017)**:


# Wykład 5
- `Konwolucja 1D` - jest efektywna, jeśli oczekujemy znalezienia interesujących cech z krótkich fragmentów całości danych, a lokalizacja tych cech nie jest dla nas bardzo ważna.
  - przykładem wykorzystania jest analiza genomu
  - jest przydatna do analizy np. danych z czujników (do szeregów czasowych)
- `Konwolucja 3D` - przydatna do analiz ruchu, danych medycznych (3D skany mózgu itp.) itp.
- `Style transfer` - transfer stylu jednego obrazu na inny obraz
  - z jednego obrazu wyciągane są obiekty, z drugiego styl, a następnie to dane są mergowane
  - w sieciach CNN początkowe warstwy dobrze radzą sobie z wyciąganiem stylu, a dalsze z wyciąganiem obiektów i ich klasyfikacją
  - co każdą epokę poprawiane są straty związane ze stylem i obiektami, po kilkudziesięciu / kilkuset epokach powinniśmy uzyskać obraz bardzo podobony treściowo i stylowo do obrazów podanych na wejście
- **`Autoencoder`** - typ sieci, który pozwala na np. odtworzenie zaszumionych zdjęć, skanowanie dokumentów czy implementacje takich aplikacji jak np. FaceSwap. `Autoencoder` składa się z enkodera i dekodera. Enkoder przetwarza wejście na formę skompresowaną, a następnie dekoder próbuje to wejście odtworzyć.
  - enkoder i dekoder są trochę lustrzanymi odbiciami: enkoder przeprowadza konwolucje i poolingi, podczas gdy dekoder dekonwolucje i unpoolingi
  - zanim sieć zostanie nakarmiona wejściem, na obraz wejściowy nakłada się trochę szumu
- `U-net` - idea podobna do autoencodera, z tą różnicą, że wyniki kompresji przesyłane są na odpowiadające jej warstwy dekompresji, co pozwala na segmentacje obrazu

# Wykład 6
Modele do klasyfikacji (np. VGG-16, ResNet itp.) nie radzą sobie dobrze dla obrazów, które zawierają wiele obiektów. Należy najpierw z obrazu wyciągnąć pojedyczne obiekty, które następnie poddawane są klasyfikacji - do tego służą **modele detekcji i segmentacji**.
- `semantic segmantation` - klasyfikacja każdego piksela obrazu do określonych klas
- `classification + localization` - klasyfikacja oraz lokalizacja obiektu na obrazie (ograniczenie do 1 obiektu)
- `object detection` - bardzie ogólny przypadek klasyfikacji i lokalizacji: wykrycie wszystkich obiektów na ekranie (znalezienie boxa dla nich) i ich klasyfikacja
- `instance segmentation` - semantyczna segmentacja, ale odróżniamy obiekty tej samej klasy od siebie

W detekcji obiektów pojawia się problem - jak ocenić skuteczność predykcji (jak porównać ze sobą boxy)? Mozna użyć **`Intersection over Union`**, który oblicza stosunek części wspólnej boxów do ich sumy.
- jest lepszy od `pixel accuracy` (procent dobrze sklasyfikowanych pikseli), m.in. dlatego, że działa lepiej dla bardzo niezbalansowanych zbiorów klas

Metody detekcji:
- `sliding method` - podział zdjęcia na komórki, a później iteracja po nich oknem i sprawdzanie zawartości. Jest to metoda wolna i potencjalnie nieskuteczna (co jeśli obiekty mogą mieć bardzo skrajne rozmiary, jaki rozmiar okna wtedy dobrać?)
- `selective search` - dzielimy obraz na regiony, które są do siebie podobne kolorem, teksturą, rozmiarem albo kształtem. Algorytm działa w sposób iteracyjny:
  1. Wygeneruj bardzo posegmentowaną reprezentację obrazu
  2. Dodaj wszystkie obliczone boxy do listy `Regions proposals`
  3. Zgrupuj sąsiadujące boxy uwzględniając ich podobieństwo
  4. Idź do 2  
  Algorytm działa bardzo szybko i generuje zdecydowanie mniej boxów od `sliding method`

- **`R-CNN` (2012)**:
  - algorytm działa w następujących krokach
    - wykorzystuje `selective search`, który odnajduje 2000 RoI (`Regions of interest` / `Region proposals`)
    - każdy region przepuszczany jest przez sieć CNN, która zwraca wektor cech
    - cechy są przekazywane do `SVM`, który klasyfikuje regiony
      - cechy ze wszystkich regionów przemnażane są przez klasy detekcji, wybierane są te, które mają `IoU` > 0.3
    - dodatkowo używany jest `Bounding box regressor`, czyli sieć która na zależnie od wykrytej klasy obiektu poprawia boxy
      - na wejście bierze lokalizacje regionu i podczas trenowania porównuje z prawdziwymi boxami
      - po trenowaniu wybierany jest odpowiedni regressor dla klasy, który mapuje lokalizację regionu
  - problemy R-CNN:
    - długio czas analizy
    - niezmienny `selective search algorithm` generujący błędne regiony
    - długi proces nauki (ponieważ wykorzystujemy wiele sieci)
- **`Fast R-CNN` (2014)**:
  - pomysł jest następujący: po co uruchamiać CNN dla każdego regionu, kiedy można uruchomić CNN dla całego obrazka, a wygenerowaną mapę cech podzielić na regiony z `salective search`?
  - wykorzystuje `RoI Pooling` - na mapę wektorów cech nakładane są regiony. Ponieważ do klasyfikacji chcemy mieć dane o tym samym rozmiarze, wszystkie regiony poolujemy do określonego rozmiaru
  - przyspieszyliśmy działanie sieci, ale tym razem to generowanie regionów zajmuje najwięcej czasu
- **`Faster R-CNN` (2015)**:
  - rezygnujemy z `selective search` na rzecz `RPN` (`Regional Proposal Network`) - sieci CNN generującej regiony korzystającej z wygenerowanej przez poprzednią sieć mapą cech:
    1. Generowane są środki `anchorów` dla obrazu (potencjalne środki boxów)
    2. Dla każdego ze środków generowane są `anchory` - kilkanaście potenchalnych boxów
    3. Nasza sieć `RPN` klasyfikuje, które regiony mają obiekt
  - może być wykorzystywana w czasie rzeczywistym
- **`YOLO` (2014)**:
  - tutaj jedna sieć oblicza boxy i prawodpodobieństwo klas dla tych boxów
  - sposób działania:
    1. Obraz dzielony jest na komórki SxS, np. 13x13
    2. Dla każdej z komórek równolegle:
      - generuje się 5 ramek, a każdej z nich przypisuje się `confidence score`
      - każda z komórek przewiduje klasę
    3. Oba wyniki łączone są w jeden - prawdopodobieństwo występowania obiektu jest przemnażane przez wszystkie prawdopdobieństwa klas dla danego boxa
    4. Zostawiamy tylko te ramki, które mają dobry wynik
   - `YOLO` jest dużo szybsze od wszystkich wariantów `R-CNN`, ale jest mniej dokładne
   - wykorzystuje **`Non-maximum suppresion`**: algorytm, który zmniejsza liczbę boxów na końcu działania
     - algorytm bierze tylko najlepsze boxy, a te, które w dużym stopniu przecinają się z nim (obliczane to jest z wykorzystaniem `IoU`) są pomijane
- `YOLOv2` (2016):
  - *Better, Faster, Stronger*
  - używa autorską sieć DarkNet
- `YOLOv3` (2018):
  - *Better, Not Faster, Stronger*
  - wylicza więcej boxów niż `YOLOv2`
- **`SSD` (2015)**:
  - podobne do `Faster R-CNN`:
    1. Konwolucja przetwarza cały obraz i generuje mapę cech
    2. Dla każdej komórki generowane są boxy
    3. Dla każdego boxu obliczane są klasy
    4. Uruchamiany jest algorytm `Non-maximum suppresion`
  - działa szybciej od `YOLO`, ale jest mniej dokładny


# Wykład 7
Sieci potrafiły malować kopie obrazów, ale nie potrafiły stworzyć obrazów od zera. Na pomoc przyszły sieci **`GAN`**. Są to sieci złożone z dwóch mniejszych sieci: **generatora i dyskryminatora**.  
- generator - generuje z szumu podróbki
- dyskryminator - próbuje rozróżnić podróbki od prawdziwych obrazów  
Obie sieci patrząc na strukturę przypominają swoje lustrzane odbicie.  

Pojawia się pytanie: jak ocenić skuteczność GAN? Można do tego wykorzystać `FID` (`Frechet Inception Distance`) - prawdziwe obrazy i podróbki są przepuszcane przez sieć `Inception`, a następnie mierzony jest dystans pomiędzy nimi: im są bliższe, tym podróbka jest lepsza.

`GAN` generuje losowe obrazy z szumu - co jeśli chcemy uzyskać obrazy z konkretnej klasy? Można do tego wykorzystać `Conditional GAN`:
- generator posiada dodatkowe wejście: klasę obiektu do wygenerowania
- dyskryminator dostaje informację o klasie obiektu przed samę oceną


# Wykład 8
## Data augmentation
Im więcej danych, tym lepiej. Gdy mamy za mało danych, można wygenerować nowe z dotychczasowych. Metoda ta nazywa się **`data augmentation`**. Jest wiele metod, np. odwrócenie obracu, rotacja, skalowanie, przycinanie, translacja, dodawania szumu, zmiana kolorów, blur, jasność, saturacja. Do bardziej wyszukanych metod można zaliczyć Style transfer, GAN, wycinanie losowych częsci obrazka (obstrukcja). Metody można łączyć ze soba.  
Dla augmentacji należy wybrać odpowiednie metody, aby odzwierciedlić scenariusze wzięte z życia. Zły wybór metod może pogorszyć dokładność modelu. Należy też pamiętać, że prawdziwe dane są zawsze lepsze od wygenerowanych.  
Nalezy pamiętać o transformacji boxów!

## Transfer learning
**`Transfer learning`** - metoda wykorzystywania nauczonych już modeli do klasyfikacji nowych danych.  
Pierwsze warstwy DNN wykrywają proste wzory, np. krawędzie. Dalsze warstwy wykrywają coraz to bardziej szczegółowe cechy. Można uciąć końcowe warstwy i w ich miejsce dodać własne - dzięki temu model się uczy dużo szybciej i działa wystarczająco dobrze nawet po nakarmieniu go małą ilością danych.  
Najczęściej podmieniamy ostatnią warstwę, ale oprócz tego można przetrenować niektóre warstwy - wtedy mówimy o `fine tuning`. Możemy wtedy zamrozić niektóre wagi. Użycie tej metody wymaga to od nas większej wiedzy o wykorzystywanym modelu.

Czasami mamy bardzo mało zdjęc dla danej klasy, albo chcemy często dodawać nowe klasy. Wtedy przyda się metoda `One shot learning` - jest to wariant `transfer learningu`, gdzie uczymy model z wykorzystaniem tylko jednego lub kilku przykładów.  
Według niektórych źródeł dotrenowywanie modelu częścią datasetu, która była przeznaczona na walidację, stanowi `Fite tuning`.

## Ataki na CNN
Jak zmylić sieć? Istnieje wiele metod ataku:
- `Fast Gradient Sign Method (FSGM)` - typ ataku, w którym złoczyńca ma pełen dostęp do modelu. Może on przygotować specjalny szum, który odkształca wejście, dzięki czemu model może być nawet w 99% pewny że na obrazku pandy widzi gibona. Niesamowite w tym rozwiązaniu jest to, że zniekształcony obraz jest nie do odróżnienia przez ludzkie oko od oryginalnego.
- `Jacobian-based saliency map (JSMA)` - tutaj zmienane jest parę pikseli obrazka,
- `One pixel attack` - a tutaj tylko jeden
- `Adversarial patch` - specjalnie przygotowane obrazki, które mogą być wykorzystane w prawdziwym życiu. Są uniwersalne (mogą być wykorzystane gdziekolwiek), nawet poddane transformacji mogą działać i można z ich wykorzystaniem targetować konkretne klasy.

Metody ochrony:
- `Adversarial training` - nauczenie się ataków, np. poprzez dodanie zaszumionych zdjęć do zbioru danych
- rozmycie, kompresja, redukcja cech - zaszumianie danych, dzięki czemu wyspecjalizowane ataki takie jak `FSGM` są nieskuteczne
- można również wykorzystać autoencoder
