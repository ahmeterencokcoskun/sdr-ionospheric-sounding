// Otomatik Üretilen Akademik Bulgular Raporu (Typst)
// Yazar: Ahmet Eren Çokcoşkun (TA7CAE) - Nuh Naci Yazgan Üniversitesi EEE

#set page(paper: "a4", margin: 2cm)
#set text(font: "New Computer Modern", size: 10.5pt, lang: "tr")

= Çok-Bantlı Fırsatçı İyonosferik Algılama ve Mikro-Doppler Kestirim Bulguları
*Operatör:* Ahmet Eren Çokcoşkun (`TA7CAE`) | *Konum:* Kayseri / Talas (`KM78`, $38.72°N, 35.48°E$)

== 1. Appleton-Hartree ve Sapmasız Yaklaşım Zayıflama ($alpha$) Hata Bütçesi
Kayseri jeomanyetik alanı ($B_0 = 48.2 space mu"T"$, $f_H approx 1.35 "MHz"$) altında elde edilen iki yönlü sönümleme karşılaştırması:

#table(
  columns: (auto, auto, auto, auto, auto, auto, auto),
  align: center,
  [*Frekans*], [*Katman*], [*İrtifa*], [*Basit $1/f^2$*], [*Tam O-Modu*], [*Tam X-Modu*], [*X-Modu Farkı*],
  [1.840 MHz], [E / Es], [95.0 km], [265.31 dB], [157.51 dB], [116.53 dB], [148.78 dB],
  [3.573 MHz], [E / Es], [96.5 km], [99.23 dB], [64.98 dB], [168.46 dB], [69.23 dB],
  [7.074 MHz], [E / Es], [100.0 km], [34.06 dB], [26.09 dB], [43.71 dB], [9.65 dB],
  [10.136 MHz], [E / Es], [103.0 km], [19.71 dB], [16.43 dB], [23.70 dB], [3.99 dB],
  [14.074 MHz], [F1], [168.5 km], [13.91 dB], [12.29 dB], [16.68 dB], [2.77 dB],
  [21.074 MHz], [F2], [233.0 km], [6.29 dB], [5.38 dB], [6.52 dB], [0.23 dB],
)

== 2. Ortak-Mod TCXO Kalibrasyonu ve Alt-Paketsel Mikro-Doppler
- *Kestirilen RSPdx TCXO Ortak-Mod Kayması:* `+0.541 Hz` (Gerçek: `+1.350 Hz`)

== 3. WSPR (4-FSK) İyonosferik Frekans Sürüklenmesi (Drift Rate) ve Katman İvmesi
#table(
  columns: (auto, auto, auto, auto, auto, auto),
  align: center,
  [*Çağrı İşareti*], [*Grid*], [*Menzil*], [*SNR*], [*İyonosferik Sürüklenme*], [*Katman İvmesi $a_z$*],
  [SIG_488Hz], [WAV], [2200 km], [+25.0 dB], [+1.14 Hz/dk], [-0.657 m/s²],
  [SIG_869Hz], [WAV], [2200 km], [+25.0 dB], [-1.27 Hz/dk], [+0.729 m/s²],
  [SIG_925Hz], [WAV], [2200 km], [+12.7 dB], [+24.40 Hz/dk], [-13.993 m/s²],
  [SIG_1369Hz], [WAV], [2200 km], [+25.0 dB], [+1.02 Hz/dk], [-0.585 m/s²],
  [SIG_1688Hz], [WAV], [2200 km], [+23.0 dB], [-7.50 Hz/dk], [+4.304 m/s²],
  [SIG_2306Hz], [WAV], [2200 km], [+25.0 dB], [-1.02 Hz/dk], [+0.585 m/s²],
)
