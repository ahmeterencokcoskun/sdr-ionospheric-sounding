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
  [1.840 MHz], [D], [89.0 km], [212.72 dB], [122.46 dB], [1313.62 dB], [1100.89 dB],
  [3.573 MHz], [D], [90.5 km], [76.08 dB], [49.23 dB], [144.34 dB], [68.27 dB],
  [7.074 MHz], [E / Es], [94.0 km], [29.77 dB], [24.86 dB], [48.57 dB], [18.80 dB],
  [10.136 MHz], [E / Es], [102.5 km], [26.30 dB], [26.81 dB], [46.67 dB], [20.37 dB],
  [14.074 MHz], [E / Es], [141.0 km], [17.73 dB], [16.53 dB], [22.85 dB], [5.11 dB],
  [21.074 MHz], [F1], [188.5 km], [7.96 dB], [6.92 dB], [8.40 dB], [0.45 dB],
  [28.074 MHz], [F2], [246.5 km], [4.52 dB], [3.92 dB], [4.53 dB], [0.02 dB],
)

== 2. Ortak-Mod TCXO Kalibrasyonu ve Alt-Paketsel Mikro-Doppler
- *Kestirilen RSPdx TCXO Ortak-Mod Kayması:* `+1.742 Hz` (Gerçek: `+1.350 Hz`)

== 3. WSPR (4-FSK) İyonosferik Frekans Sürüklenmesi (Drift Rate) ve Katman İvmesi
#table(
  columns: (auto, auto, auto, auto, auto, auto),
  align: center,
  [*Çağrı İşareti*], [*Grid*], [*Menzil*], [*SNR*], [*İyonosferik Sürüklenme*], [*Katman İvmesi $a_z$*],
  [F1GJD], [BF23], [3539 km], [+8.3 dB], [-0.00 Hz/dk], [+0.001 m/s²],
  [VK7ZNP], [JH77], [1536 km], [+11.8 dB], [-0.86 Hz/dk], [+0.494 m/s²],
  [PY1DBQ], [KG18], [2972 km], [+12.9 dB], [+0.16 Hz/dk], [-0.090 m/s²],
  [G2HUE], [BM88], [3502 km], [+6.4 dB], [-0.41 Hz/dk], [+0.237 m/s²],
)
