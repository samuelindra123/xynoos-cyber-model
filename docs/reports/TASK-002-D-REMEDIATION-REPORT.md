# TASK 002-D — Laporan remediasi numerik

## Status dan cakupan

Remediasi Session B siap diperiksa ulang oleh Session A melalui draft PR #2. Audit TASK 002-C menemukan satu masalah implementasi yang terkonfirmasi, C-001, dan dua celah validasi pada T-15 dan T-16. Perubahan ini menutup C-001 untuk T-05 dan T-06, membuktikan penolakan input fixture yang salah untuk T-15, dan menetapkan versi Rust yang dapat dipasang ulang untuk T-16. [Audit dan traceability](https://github.com/samuelindra123/xynoos-cyber-model/tree/51c4d1cf2faa3b3b639c3e18d6fa79fc8a5515b6/docs/reports) · [Handoff audit](https://github.com/samuelindra123/xynoos-cyber-model/blob/9f9dd8a2750e74e196feeac8defde4bd2f3d78ef/docs/design/TASK-002-D-REMEDIATION-HANDOFF.md) · [PR #2](https://github.com/samuelindra123/xynoos-cyber-model/pull/2)

Pekerjaan dilanjutkan dari `70045ba2dfdd65ad9edbe7ba3122d815d28c914a` di `feat/task-002-b-numerical-foundation`. Implementasi C-001 dan sembilan tes kapasitas sudah tersedia di `56c874b90374804e9b34faecb10d36f3dda386fc`. CI pada head awal lulus. Commit `8a3719f9ba8d0b9c847a87189b97a96318d893e5` menambahkan validasi batas proyek, kontrol negatif, pin toolchain, dan pembaruan kontrak. [CI head awal](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45530208545)

## C-001 dan kontrak kapasitas

Pilihan C-001 adalah menolak kapasitas berlebih. `TensorLimits` membatasi `len * size_of::<f32>()` dan `capacity * size_of::<f32>()` hingga 64 MiB per tensor, dengan opsi batas lebih rendah. `from_vec` memvalidasi payload dan panjang, lalu memeriksa kapasitas sebelum menyalin shape. Buffer yang lolos tetap memakai alokasi asli dan boleh menyimpan kapasitas cadangan dalam batas. Kegagalan mengonsumsi dan melepas `Vec` yang diserahkan. Pilihan ini menghindari salinan nilai dan membuat penolakan teramati melalui `RetainedCapacityLimitExceeded { bytes, limit }`. Perkalian kapasitas yang meluap menghasilkan `RetainedCapacityByteCountOverflow`. Buffer baru yang dibuat melalui `try_reserve_exact` diperiksa berdasarkan kapasitas yang dilaporkan sebelum nilai ditulis. [Kontrak API](../../crates/xynoos-tensor/README.md) · [Kebijakan komputasi](../COMPUTE_POLICY.md)

Sepuluh tes kapasitas lulus di kedua profil. Cakupannya meliputi adopsi tanpa salin dengan pemeriksaan pointer, kapasitas tepat pada dan di atas anggaran, vektor kosong dengan dan tanpa kapasitas pada anggaran nol, scalar, prioritas kesalahan panjang, overflow aritmetis, serta kapasitas konstruktor dan hasil operasi. Tes tambahan menerima hitungan kapasitas tepat pada 64 MiB dan menolak satu `f32` lebih banyak melalui pemeriksaan aritmetis. Tes ini juga memeriksa bahwa permintaan di atas plafon disetarakan dengan batas default. Kasus batas besar memakai hitungan sintetis, sedangkan kasus adopsi memakai vektor kecil.

## Bukti oracle

Python 3.12.3 pada CI meregenerasi fixture byte demi byte dengan seed `0x58594e4f`, 295 kasus, dan 2.670 nilai keluaran. SHA-256 fixture adalah `cdfde7ba9f44f46ba7e757157019730e1124bcd966e796b73286bddf3894fa0f`. Perbandingan finite memakai `atol=1e-6` dan `rtol=1e-5`, NaN dibandingkan berdasarkan klasifikasi, dan infinity berdasarkan tanda.

Pembaca fixture Rust yang sama menjalankan fixture lengkap, kontrol positif `1 + 2 = 3`, dan enam variasi independen atas satu rekaman di memori. Ketika input ditolak, harness mencocokkan diagnostik yang diharapkan. Keenam diagnostik berikut muncul di log debug dan release. [Log job](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45540891961/job/138958846528)

| Mutasi rekaman | Diagnostik yang teramati |
|---|---|
| Hapus field terakhir | `line 2: invalid fixture record` |
| Ganti operasi `add` dengan `unknown` | `line 2: unknown operation unknown` |
| Ganti word keluaran dengan `not-hex` | `invalid binary32 fixture value` dengan `InvalidDigit` |
| Ganti shape keluaran scalar `[]` dengan `[1]` | `line 2: add output shape` |
| Sisipkan nilai keluaran kedua | `line 2: add output count` |
| Ganti nilai harapan 3 dengan 4 | `line 2: add output 0: expected 4.0, got 3.0` |

## Toolchain dan hasil GitHub Actions

`rust-toolchain.toml` menetapkan Rust 1.100.0, profil minimal, rustfmt dan Clippy. Versi ini sama dengan rilis yang dipakai oleh validasi sebelumnya. CI memakai pin repository, memverifikasi versi `rustc`, mencatat versi alat, dan menjalankan Cargo dengan lockfile secara offline setelah memasang toolchain. Workflow memakai satu job GitHub-hosted Ubuntu 24.04, timeout 15 menit, maksimal dua job build Cargo, izin `contents: read`, dan pembatalan run yang tergantikan.

Run `45540891961` menyelesaikan seluruh langkah dengan sukses. Log checkout mencatat `8a3719f9ba8d0b9c847a87189b97a96318d893e5` dan `rustc 1.100.0 (0ef1bc386 2026-11-09)`. Kolom hasil di bawah merujuk ke run tersebut. SHA head final dan CI yang menguji commit laporan ini dicatat di deskripsi PR #2 setelah verifikasi. [Run validasi](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45540891961) · [Pemeriksaan head PR #2](https://github.com/samuelindra123/xynoos-cyber-model/pull/2/checks)

| Pemeriksaan | Hasil |
|---|---|
| Revisi checkout, berkas wajib, Markdown, dan YAML | Lulus |
| Kebijakan privasi dan artefak | Lulus untuk 44 berkas terlacak |
| Pin Rust, `cargo metadata --locked --offline`, dan `cargo fmt --all -- --check` | Lulus |
| `cargo clippy --workspace --all-targets --locked --offline -- -D warnings` | Lulus |
| `python3 tools/numerical_oracle.py --check` | Lulus, 295 kasus dan 2.670 nilai |
| `cargo test --workspace --locked --offline -- --show-output` | Lulus, 13 unit, 10 kontrak/properti, 8 oracle, 1 doctest |
| `cargo test --workspace --release --locked --offline -- --show-output` | Lulus, 13 unit, 10 kontrak/properti, 8 oracle, 1 doctest |
| `RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps --locked --offline` | Lulus |

## Batas yang masih berlaku

Anggaran tensor meliputi kapasitas numerik yang dilaporkan `Vec`. Metadata, overhead dan pembulatan yang tersembunyi pada allocator, agregat tensor hidup, RSS, dan OS overcommit berada di luar anggaran. Tes sengaja memakai input kecil dan overflow sintetis. Biaya `matmul` dapat mencapai O(mkn), dengan pembatasan kerja tetap menjadi tanggung jawab pemanggil.

Oracle memberi bukti atas fixture terpilih dan jalur penolakan pembaca yang diuji. `math.fsum` pada oracle dan akumulasi binary64 berurutan pada Rust dapat berbeda untuk pembatalan ekstrem. Pengujian ini tidak menetapkan akurasi universal atau kesetaraan bit lintas target. Pin Rust mencakup toolchain. Versi Python dan image runner dapat bergerak, dan pemindaian privasi tetap bersifat heuristik.

## Handoff verifikasi ulang Session A

1. Ambil SHA head dan URL CI dari deskripsi PR #2. Cocokkan SHA dengan `headRefOid`, metadata `headSha` run dan baris `Tested commit` di job. Pastikan run dan semua langkah wajib berstatus `success`.
2. Tinjau `Tensor::from_vec`, `collect_data`, dan dua varian error kapasitas. Cocokkan urutan validasi, pemeriksaan kapasitas sebelum menulis, adopsi pointer, kasus anggaran nol, serta batas sintetis 64 MiB dengan kontrak tertulis. [Kode tensor](../../crates/xynoos-tensor/src/tensor.rs) · [Error bertipe](../../crates/xynoos-tensor/src/error.rs)
3. Cocokkan hasil kontrol positif, enam mutasi dan seed dengan log debug dan release. Jalankan `python3 tools/numerical_oracle.py --check` dan `cargo test -p xynoos-tensor --test numerical_oracle --locked --offline -j 1 -- --show-output` bila ingin mereproduksi diagnostik secara lokal. [Pembaca fixture dan kontrol](../../crates/xynoos-tensor/tests/numerical_oracle.rs)
4. Cocokkan toolchain yang tercatat dengan `rust-toolchain.toml` dan tinjau batas job, izin, serta perintah Cargo offline. Catat keputusan independen untuk C-001 dan T-15/T-16 pada SHA yang diperiksa. Gunakan checkout kanonis dengan kebijakan satu worktree. [Pin toolchain](../../rust-toolchain.toml) · [Workflow](../../.github/workflows/ci.yml) · [Kebijakan workspace](../../AGENTS.md)
