# Midterm Project Checklist (Speech Analysis & Gender Classification)

## 1) Teslimat ve Genel Gereksinimler
- [ ] Kod dili Python olacak.
- [ ] Kodlar GitHub reposuna konulacak.
- [ ] Rapor adı: `MidtermProject_GRUP#.pdf`
- [ ] Sunum adı: `MidtermProject__Presentation_GRUP#`
- [ ] Teslim Blackboard üzerinden yapılacak.

## 2) Veri Seti ve Metadata
- [x] Tüm grup klasorleri tek bir ana klasorde toplanacak (`Dataset/Group_01`, `Group_02`, ...).
- [x] Tum Excel metadata dosyalari okunup tek bir master metadata dosyasinda birlestirilecek.
- [x] `.wav` dosyalari dongu ile taranip dosya-varlik kontrolu yapilacak.
- [x] Eksik / hatali dosyalar raporlanacak.

## 3) Zaman Bolgesi Analizi (Time Domain)
- [x] Sinyal 20-30 ms pencerelere bolunecek.
- [x] Short-Term Energy (STE) hesaplanacak.
- [x] Zero Crossing Rate (ZCR) hesaplanacak.
- [x] STE + ZCR ile voiced bolgeler secilecek.

## 4) F0 (Pitch) Cikarimi - Autocorrelation
- [x] Autocorrelation fonksiyonu uygulanacak.
- [x] F0 yalnizca voiced pencerelerde hesaplanacak.
- [x] Her kayit icin ortalama F0 uretilecek.
- [ ] Bir secili ornek icin Autocorrelation ve FFT yan yana gorsellestirilecek.

## 5) Ozellik Cikarimi ve Siniflandirma
- [x] Her ses dosyasi icin ozellikler: Ortalama F0, Ortalama ZCR, Enerji olcumu.
- [x] Rule-based siniflandirici (Male/Female/Child) tasarlanacak.
- [x] Tum veri setinde tahmin yapilip dogruluk (accuracy) hesaplanacak.
- [x] Confusion matrix olusturulacak.

## 6) Istatistik ve Hata Analizi
- [x] Sinif bazli tablo: ornek sayisi, ortalama F0, std sapma, basari yuzdesi.
- [ ] Yanlis siniflandirmalar teknik olarak yorumlanacak (gurultu, ton, duygu durumu, vb.).

## 7) Arayuz (UI)
- [ ] Arayuz secilecek: Jupyter / Tkinter / PyQt / Streamlit / Flask.
- [ ] Kullanici secilen ses dosyasini siniflandirabilecek.
- [ ] Tum veri seti performansi arayuzde veya raporda gosterilecek.

## 8) Rapor Icerigi
- [ ] Giris (amac ve kapsam)
- [ ] Veri seti karakterizasyonu
- [ ] Metodoloji (STE, ZCR, autocorrelation)
- [ ] Autocorrelation vs FFT karsilastirma
- [ ] Istatistiksel bulgular tablosu
- [ ] Siniflandirma basarisi + confusion matrix
- [ ] Hata analizi ve tartisma
- [ ] GitHub linki
- [ ] References
- [ ] AI prompt tablosu (kullanildiysa)
- [ ] Is bolumu (takim katkilari)

## 9) Sunum (7 Dakika)
- [ ] Kapak ve giris
- [ ] Veri seti ozeti
- [ ] Yontem
- [ ] Bulgular ve karsilastirma
- [ ] Canli demo
- [ ] Hata analizi ve kapanis

---

## Adim Adim Uygulama Plani
1. [x] Proje klasor yapisini olustur (`src/`, `notebooks/`, `data/`, `outputs/`).
2. [x] Metadata birlestirme scriptini yaz (`src/build_metadata.py`).
3. [x] Ozellik cikarma scriptini yaz (`src/extract_features.py`).
4. [x] Rule-based siniflandiriciyi yaz (`src/classifier.py`).
5. [x] Degerlendirme scriptini yaz (`src/evaluate.py`).
6. [ ] Ornek UI/notebook olustur (`notebooks/demo.ipynb` veya Streamlit).
7. [x] Sonuclarla rapor tablolari ve confusion matrix ciktisi al.

> Not: Dataset guncellendikten sonra pipeline yeniden calistirildi. Yeni sonuclar: `Accuracy=%69.85`, `N=262`.
