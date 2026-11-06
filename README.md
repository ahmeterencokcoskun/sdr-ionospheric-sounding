<div align="center">

# `sdr-ionospheric-sounding` // IONO-SENSE TA7CAE
### **HF Sayısal Modları ve SDR Kullanılarak Çok-Bantlı Fırsatçı İyonosferik Algılama: Katman Dinamikleri ve Mikro-Doppler Kestirimi**

[![Operator](https://img.shields.io/badge/Operatör-TA7CAE%20(Ahmet%20Eren%20Çokcoşkun)-critical?style=flat-square&logo=rss&logoColor=white)](https://github.com/ahmeterencokcoskun)
[![Institution](https://img.shields.io/badge/Kurum-Nuh%20Naci%20Yazgan%20Üniversitesi%20EEE-blue?style=flat-square)](#)
[![QTH](https://img.shields.io/badge/QTH-Kayseri%20%2F%20Talas%20(38.72°N%2C%2035.48°E%20—%20KM78)-success?style=flat-square)](#)
[![Hardware](https://img.shields.io/badge/SDR-SDRplay%20RSPdx%20(14--bit%20ADC)-informational?style=flat-square)](#)
[![Standards](https://img.shields.io/badge/ITU--R-P.533%20%7C%20F.1487-orange?style=flat-square)](#)

</div>

---

## 1. Proje Özeti ve Sistem Mimarisi

Bu depo; **Nuh Naci Yazgan Üniversitesi Elektrik-Elektronik Mühendisliği Bölümü** Lisans Bitirme Tezi kapsamında geliştirilen, küresel ölçekte atomik saat senkronizasyonuyla yayın yapan fırsatçı HF sayısal sinyallerini (**`FT8` 8-GFSK** ve **`WSPR` 4-FSK**) pasif bistatik bir iyonosond gibi kullanarak orta enlem Anadolu iyonosferinin (**D, E, E<sub>s</sub>, F<sub>1</sub>, F<sub>2</sub>**) katman dinamiklerini, magneto-iyonik zayıflama hata bütçesini ve alt-paketsel mikro-Doppler yörüngelerini kestiren uçtan uca sinyal işleme (DSP) ve plazma fiziği motorunu içerir.

<div align="center">
  <img src="iono_sense_dashboard.png" alt="IONO-SENSE TA7CAE 6-Panel Akademik Analiz Paneli" width="100%" />
</div>

---

## 2. Teorik Çerçeve ve Matematiksel Formülasyon

### A. Magneto-İyonik Zayıflama (α) ve "Yaklaşım vs. Tam Fizik" Hata Bütçesi (`plasma_physics.py`)
Literatürdeki standart sapmasız (non-deviative) `1/f²` yaklaşımı ile Kayseri jeomanyetik alanı (`B₀ = 48.2 µT`, manyetik eğim açısı `I = 56.4°`, elektron jiroskop frekansı `f_H ≈ 1.35 MHz`) altında çözülen tam **Appleton-Hartree** ve kinetik **Sen-Wyller** modelleri arasındaki farkı (dB Discrepancy) hesaplar:

```math
\tilde{n}^2 = (\mu - j\chi)^2 = 1 - \frac{X}{1 - jZ - \frac{Y_T^2}{2(1 - X - jZ)} \pm \sqrt{\frac{Y_T^4}{4(1 - X - jZ)^2} + Y_L^2}}, \qquad \alpha_{O,X}(h) = \frac{\omega}{c}\left|\mathrm{Im}(\tilde{n}_{O,X})\right|
```

#### Tablo 1: Kayseri (`KM78`) Orta Enlem İyonosferinde Katman Etkileşimi ve Zayıflama Hata Bütçesi (Kalkış Açısı Δ = 18°)

| Taşıyıcı Frekans | Yansıyan Katman | Kırılma İrtifası (h<sub>refl</sub>) | Basit `1/f²` Yaklaşımı | Tam O-Modu | Tam X-Modu (Jiro-Rezonans) | Model Hata Payı |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1.840 MHz`** (160m) | D | `83.5 km` | `243.88 dB` | `139.66 dB` | `1108.63 dB` | **`864.75 dB`** |
| **`3.573 MHz`** (80m) | D | `88.0 km` | `101.73 dB` | `68.43 dB` | `201.07 dB` | **`99.35 dB`** |
| **`7.074 MHz`** (40m) | E / E<sub>s</sub> | `94.5 km` | `39.34 dB` | `30.90 dB` | `55.19 dB` | **`15.84 dB`** |
| **`10.136 MHz`** (30m) | E / E<sub>s</sub> | `101.5 km` | `24.45 dB` | `20.51 dB` | `30.99 dB` | **`6.54 dB`** |
| **`14.074 MHz`** (20m) | F<sub>1</sub> / F<sub>2</sub> | `154.5 km` | `15.16 dB` | `13.32 dB` | `17.97 dB` | **`2.81 dB`** |
| **`21.074 MHz`** (15m) | F<sub>1</sub> / F<sub>2</sub> | `207.0 km` | `6.81 dB` | `6.22 dB` | `7.60 dB` | **`0.79 dB`** |
| **`28.074 MHz`** (10m) | F<sub>2</sub> | `236.0 km` | `3.85 dB` | `3.59 dB` | `4.18 dB` | **`0.32 dB`** |
| **`50.313 MHz`** (6m) | **Uzaya Kaçış** | `---` | `1.24 dB` | `1.19 dB` | `1.29 dB` | **`0.06 dB`** |

---

### B. Zamanla Değişen Watterson Kanalı ve Alt-Paketsel Mikro-Doppler Kestirimi (`watterson_simulator.py` & `dsp_core.py`)
Çok-yollu (`1F2` ve `2F2`) gökyüzü dalgası yayılımı **ITU-R F.1487 Watterson** modeliyle temsil edilir:

```math
h(\tau, t) = \sum_{m=1}^{M} G_m(t) \, e^{j 2\pi f_{d,m}(t) t} \, \delta(\tau - \tau_m(t))
```

Alıcı yerel osilatörünün (RSPdx TCXO) termal kayması, aynı 15 saniyelik pencerede çözümlenen istasyonların ortak-mod medyan kestiricisiyle sıfırlanır; modülasyon fazı soyulduktan sonra Savitzky-Golay türev filtresiyle anlık mikro-Doppler yörüngesi ve katman dikey plazma hareket hızı çıkarılır:

```math
f_d(t) = \frac{1}{2\pi} \frac{d}{dt}\left[\mathrm{unwrap}\left(\arctan\frac{Q(t)}{I(t)}\right)\right] - \Delta \hat{f}_{\mathrm{common}}, \qquad v_z(t) = -\frac{c}{2 f_c \cos\theta_i} \, f_d(t)
```

#### Tablo 2: `14.074 MHz` (20m F<sub>2</sub> Katmanı) FT8 Çözümleme, TCXO Kalibrasyonu ve Dikey Plazma Hızı Çıktıları
*(Gerçek RSPdx TCXO Kayması: `+1.350 Hz` → Kestirilen Ortak-Mod Kayması: `+1.526 Hz`)*

| Çağrı İşareti | Grid | Menzil | SNR (2500 Hz) | Kalibre Edilmiş Doppler (f<sub>d</sub>) | Doppler Yayılması (σ<sub>D</sub>) | Dikey Katman Hızı (v<sub>z</sub>) | Pre-FEC BER |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`DL1ABC`** | `JO52` | `2180 km` | `+8.5 dB` | `+0.245 Hz` | `0.234 Hz` | `-8.08 m/s` | `0.0000 (0/174)` |
| **`G4XYZ`** | `IO91` | `2840 km` | `+4.0 dB` | `-0.498 Hz` | `0.378 Hz` | `+14.57 m/s` | `0.0000 (0/174)` |
| **`I2QRS`** | `JN45` | `1920 km` | `+11.2 dB` | `+0.000 Hz` | `0.194 Hz` | `+0.39 m/s` | `0.0000 (0/174)` |
| **`OH2MNO`** | `KP20` | `2410 km` | `-2.5 dB` | `-0.842 Hz` | `0.417 Hz` | `+29.44 m/s` | `0.0000 (0/174)` |
| **`EA4DEF`** | `IN80` | `3310 km` | `+1.5 dB` | `+0.314 Hz` | `0.336 Hz` | `-13.50 m/s` | `0.0000 (0/174)` |

---

## 3. Depo Dosya Yapısı

```text
sdr-ionospheric-sounding/
├── plasma_physics.py          # Chapman Ne(h), Appleton-Hartree (O/X-Mode), Sen-Wyller ve Ray-Tracing çözücüsü
├── watterson_simulator.py     # ITU-R F.1487 Watterson çok-yollu kanal ve 8-GFSK FT8 / LDPC(174,91) I/Q üreticisi
├── dsp_core.py                # FIR DDC kanal izolasyonu, Costas 2B uyumlu filtre, faz açma ve BER analizörü
├── sdr_hardware.py            # SoapySDR (SDRplay RSPdx) donanım soyutlama katmanı, bant atlama ve SQLite kaydedici
├── main.py                    # Ana simülasyon/sondaj çalıştırıcısı, 6-panelli çizici ve Typst/LaTeX rapor üreticisi
├── requirements.txt           # Python bilimsel hesaplama bağımlılıkları (NumPy, SciPy, Matplotlib)
├── iono_sense_dashboard.png   # Üretilen 6-panelli yüksek çözünürlüklü analiz figürü (PNG)
├── iono_sense_dashboard.pdf   # Vektörel yayın kalitesinde analiz figürü (PDF)
├── tez_bulgulari.typ          # Otomatik üretilen Typst akademik bulgular raporu
└── tez_bulgulari.tex          # Otomatik üretilen LaTeX (IEEE SİU / Bitirme Tezi) tablosu
```

---

## 4. Kurulum ve Çalıştırma

```bash
git clone https://github.com/ahmeterencokcoskun/sdr-ionospheric-sounding.git
cd sdr-ionospheric-sounding
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

---

<div align="center">

**Geliştirici:** Ahmet Eren Çokcoşkun (`TA7CAE` / `YM7XNN`) — *Nuh Naci Yazgan Üniversitesi, Elektrik-Elektronik Mühendisliği Bölümü*

</div>
