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
- *Kestirilen RSPdx TCXO Ortak-Mod Kayması:* `+1.393 Hz` (Gerçek: `+1.350 Hz`)

== 3. WSPR (4-FSK) İyonosferik Frekans Sürüklenmesi (Drift Rate) ve Katman İvmesi
#table(
  columns: (auto, auto, auto, auto, auto, auto),
  align: center,
  [*Çağrı İşareti*], [*Grid*], [*Menzil*], [*SNR*], [*İyonosferik Sürüklenme*], [*Katman İvmesi $a_z$*],
  [G5FFR], [AA53], [3076 km], [-0.5 dB], [-0.55 Hz/dk], [+0.315 m/s²],
  [DL5WML], [FM94], [1141 km], [-2.8 dB], [+0.30 Hz/dk], [-0.172 m/s²],
  [DL1VHG], [DN40], [2145 km], [-3.8 dB], [+0.20 Hz/dk], [-0.115 m/s²],
  [JA9HIM], [DL21], [4482 km], [+7.6 dB], [-0.20 Hz/dk], [+0.115 m/s²],
)

== 4. Breit-Tuve Sanal İrtifadan Gerçek İrtifa Tersinimi (True-Height Inversion)
FMCW dikey iyonosond sanal irtifa verileri $h'(f)$, Breit-Tuve integral denklemine dayanmaktadır:
$ h'(f) = h_0 + integral_0^{h_r(f)} mu'(f, f_p(h)) dif h $

Bu Abel-tipi integral denklemi, Jackson (1969) parçalı doğrusal laminasyon matris çekirdeği $M_(i k)$ ve Tikhonov 2. derece ayrık eğrilik regülarizasyon operatörü ($L$) kullanılarak tersinmiştir. Elde edilen tepe parametreleri ve alt-iyonosferik toplam elektron içeriği aşağıda özetlenmiştir:

#table(
  columns: (auto, auto, auto, auto, auto),
  align: center,
  [*foF2 (Kritik Frekans)*], [*hmF2 (Tepe İrtifası)*], [*NmF2 (Elektron Yoğunluğu)*], [*Alt-Tepe TEC*], [*Kalıntı RMS*],
  [10.871 MHz], [262.4 km], [1.466e+12 el/m³], [10.42 TECU], [0.0019 km],
)

*Tersinim ile Kestirilen Katman Profili Örneklemleri:*
#table(
  columns: (auto, auto, auto),
  align: center,
  [*Plazma Frekansı $f_p$*], [*Gerçek İrtifa $h$*], [*Elektron Yoğunluğu $N_e$*],
  [1.00 MHz], [79.0 km], [1.240e+10 el/m³],
  [2.41 MHz], [93.2 km], [7.205e+10 el/m³],
  [3.82 MHz], [98.8 km], [1.810e+11 el/m³],
  [5.23 MHz], [143.7 km], [3.393e+11 el/m³],
  [6.64 MHz], [195.2 km], [5.470e+11 el/m³],
  [8.05 MHz], [212.6 km], [8.039e+11 el/m³],
  [9.46 MHz], [230.4 km], [1.110e+12 el/m³],
  [10.87 MHz], [262.4 km], [1.466e+12 el/m³],
)
