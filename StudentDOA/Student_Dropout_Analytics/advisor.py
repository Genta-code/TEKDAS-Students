"""LLM Academic Advisor: Menerjemahkan output ML dan rekam akademik menjadi saran intervensi terstruktur.

Penggunaan mandiri:
    ollama run llama3.2
    python advisor.py
"""
import os
from typing import Any, Dict, Iterable, Optional
import ollama

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


def _compact_dict(data: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    return {k: data.get(k) for k in keys if k in data}


def build_prompt(
    student_profile: Dict[str, Any],
    dropout_probability: float,
    academic_factors: Optional[list] = None,
) -> str:
    """Membangun prompt terstruktur yang grounded pada data mahasiswa."""
    profile = _compact_dict(student_profile, [
        "Student_ID", "Course", "Age at enrollment", "Gender",
        "Tuition fees up to date", "Debtor", "Scholarship holder",
        "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (approved)", "Curricular units 2nd sem (grade)",
        "total_approved_units", "approval_rate_1st_sem", "approval_rate_2nd_sem",
        "Target",
    ])

    return f"""
Anda adalah Konselor Akademik & Analis Institusi Pendidikan Berbasis AI (Academic Success Advisor).
Gunakan HANYA fakta data mahasiswa dan output model Machine Learning berikut.
Jangan mengarang data atau membuat asumsi tanpa bukti.

PROFIL MAHASISWA:
{profile}

OUTPUT MACHINE LEARNING:
Prediksi Probabilitas Risiko Dropout: {dropout_probability:.1%}

FAKTOR GLOBAL MODEL PALING BERPENGARUH:
{academic_factors or 'Tidak tersedia.'}

TUGAS:
Tulis analisis dalam Bahasa Indonesia untuk Tim Dosen Pembimbing Akademik / Kemahasiswaan:
1. **Interpretasi Risiko**: Rangkum dalam 1-2 kalimat mengenai profil risiko mahasiswa berdasarkan probabilitas model dan capaian semester 1 & 2.
2. **Akar Permasalahan Utama (Evidence-Based)**: Sebutkan 2 faktor terkuat dari data (misal: tunggakan SPP, jumlah SKS lulus semester 2 yang rendah, atau penurunan nilai).
3. **Rencana Intervensi Akademik**: Berikan 3 langkah taktis dan konkret yang harus diambil pihak kampus (misal: bimbingan konseling, program remediasi mata kuliah, bantuan finansial/keringanan SPP).
"""


def generate_student_strategy(
    student_profile: Dict[str, Any],
    dropout_probability: float,
    academic_factors: Optional[list] = None,
    model_name: str = DEFAULT_MODEL,
) -> str:
    """Memanggil Ollama API lokal dengan fallback analisis cerdas jika Ollama belum aktif."""
    prompt = build_prompt(student_profile, dropout_probability, academic_factors)

    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]
    except Exception as exc:
        # Fallback analitik berbasis aturan jika Ollama lokal belum berjalan
        risk_level = "SANGAT TINGGI" if dropout_probability >= 0.65 else ("SEDANG" if dropout_probability >= 0.40 else "RENDAH")
        tuition_status = "Lunas" if student_profile.get("Tuition fees up to date") == 1 else "Menunggak"
        debtor_status = "Ya" if student_profile.get("Debtor") == 1 else "Tidak"
        appr_sem2 = student_profile.get("Curricular units 2nd sem (approved)", 0)
        grade_sem2 = student_profile.get("Curricular units 2nd sem (grade)", 0.0)

        fallback_output = f"""
> [!NOTE]
> *Layanan Ollama lokal ({model_name}) tidak merespons atau belum aktif (`{exc}`). Menampilkan analisis inferensi terstruktur berbasis data:*

### 🎓 Laporan Rencana Aksi Akademik — {student_profile.get('Student_ID', 'N/A')}

#### 1. Interpretasi Risiko
Mahasiswa memiliki tingkat risiko **{risk_level}** dengan probabilitas putus studi sebesar **{dropout_probability:.1%}**. Evaluasi menunjukkan performa semester kedua berada pada kelulusan **{appr_sem2} unit matkul** dengan rata-rata nilai **{grade_sem2:.2f}**.

#### 2. Akar Masalah Utama (Berdasarkan Data)
- **Kondisi Finansial**: Status pelunasan SPP: **{tuition_status}**, Status hutang/debitur: **{debtor_status}**. {'Tunggakan biaya menjadi salah satu pemicu utama stres akademik dan risiko dropout.' if tuition_status == 'Menunggak' or debtor_status == 'Ya' else 'Kondisi finansial terpantau tertib.'}
- **Retensi Akademik**: Capaian kelulusan mata kuliah semester 2 {'mengalami penurunan signifikan dan butuh perhatian segera.' if appr_sem2 < 4 else 'masih dalam rentang wajar namun memerlukan pendampingan berkelanjutan.'}

#### 3. Rekomendasi Tindakan Taktis Dosen Wali & Institusi
1. **Panggilan Konseling Akademik Segera**: Jadwalkan sesi 1-on-1 dengan Dosen Pembimbing Akademik untuk memetakan kendala pada mata kuliah yang belum tuntas.
2. **Dukungan Finansial & Administrasi**: {'Hubungkan mahasiswa dengan bagian beasiswa / fasilitasi cicilan SPP agar tidak terkena pemblokiran KRS.' if tuition_status == 'Menunggak' else 'Pertahankan skema bantuan akademik atau dorong pengajuan beasiswa prestasi.'}
3. **Program Remediasi & Peer Tutoring**: Daftarkan mahasiswa ke klinik belajar atau kelompok tutor sebaya khusus untuk mata kuliah prasyarat semester berikutnya.
"""
        return fallback_output


if __name__ == "__main__":
    test_student = {
        "Student_ID": "STD_0001",
        "Course": 2,
        "Tuition fees up to date": 0,
        "Debtor": 1,
        "Curricular units 2nd sem (approved)": 1,
        "Curricular units 2nd sem (grade)": 10.5,
    }
    print(generate_student_strategy(test_student, 0.78))
