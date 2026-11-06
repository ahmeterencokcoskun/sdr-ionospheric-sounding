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
  [1.840 MHz], [D], [83.5 km], [243.88 dB], [139.66 dB], [1108.63 dB], [864.75 dB],
  [3.573 MHz], [D], [88.0 km], [101.73 dB], [68.43 dB], [201.07 dB], [99.35 dB],
  [7.074 MHz], [E / Es], [94.5 km], [39.34 dB], [30.90 dB], [55.19 dB], [15.84 dB],
  [10.136 MHz], [E / Es], [101.5 km], [24.45 dB], [20.51 dB], [30.99 dB], [6.54 dB],
  [14.074 MHz], [F1], [154.5 km], [15.16 dB], [13.32 dB], [17.97 dB], [2.81 dB],
  [21.074 MHz], [F1], [207.0 km], [6.81 dB], [6.22 dB], [7.60 dB], [0.79 dB],
  [28.074 MHz], [F2], [236.0 km], [3.85 dB], [3.59 dB], [4.18 dB], [0.32 dB],
)

== 2. Ortak-Mod TCXO Kalibrasyonu ve Alt-Paketsel Mikro-Doppler
- *Kestirilen RSPdx TCXO Ortak-Mod Kayması:* `+1.526 Hz` (Gerçek: `+1.350 Hz`)

== 3. WSPR (4-FSK) İyonosferik Frekans Sürüklenmesi (Drift Rate) ve Katman İvmesi
#table(
  columns: (auto, auto, auto, auto, auto, auto),
  align: center,
  [*Çağrı İşareti*], [*Grid*], [*Menzil*], [*SNR*], [*İyonosferik Sürüklenme*], [*Katman İvmesi $a_z$*],
  [DP0GVN], [IB59], [13420 km], [-21.0 dB], [-0.83 Hz/dk], [+0.479 m/s²],
  [ZL1SIX], [RF73], [16890 km], [-25.0 dB], [+1.15 Hz/dk], [-0.662 m/s²],
  [K3LR], [EN91], [8940 km], [-14.0 dB], [+0.30 Hz/dk], [-0.173 m/s²],
  [JA1ABC], [PM95], [8720 km], [-18.5 dB], [-0.64 Hz/dk], [+0.369 m/s²],
)
