"""
Evaluation script: English vs Indonesian vs Chinese tokenization bias.
Large test set (~50+ sentences per language).
"""

import argparse
from transformers import AutoTokenizer

from gatoken import StandardTokenizer, RotorSubwordTokenizer, compute_metrics, compare_languages


# Large test set: English, Indonesian, Chinese
# Focused on Southeast Asian + Chinese language characteristics

ENGLISH_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "I really like eating spicy food at the night market.",
    "She went to the store to buy some fresh vegetables.",
    "The cat is sleeping on the mat.",
    "He drinks coffee every morning.",
    "My grandmother makes the best fried rice in the village.",
    "We will meet at the train station tomorrow morning.",
    "He forgot to bring his umbrella when it started raining.",
    "My sister bought a new phone yesterday.",
    "They are playing football in the field.",
    "I usually wake up at six in the morning.",
    "She cooked dinner for the whole family.",
    "Where did you go last weekend?",
    "How much does this shirt cost?",
    "Can you help me with this problem?",
    "What time does the bus arrive?",
    "The government announced new regulations for online businesses last week.",
    "I need to finish my homework before going to the cinema with friends.",
    "The old man walks slowly along the river every evening.",
    "Many tourists visit Bali during the dry season.",
    "The teacher explained the lesson very clearly today.",
    "We bought some fresh fruits and vegetables from the market.",
    "My father works as a teacher in a small school.",
    "The children are very happy playing in the park.",
    "The president will deliver a speech at the national conference.",
    "Several companies have invested in renewable energy projects.",
    "The new law aims to protect consumers from fraud.",
    "Scientists discovered a new species in the rainforest.",
    "The economy grew by five percent last year.",
    "I installed a new application on my laptop yesterday.",
    "Artificial intelligence is changing many industries rapidly.",
    "Please send the report before the deadline tomorrow.",
    "The internet connection is very slow today.",
    "She works remotely from a small town in Java.",
    "We celebrate Eid by visiting relatives and eating together.",
    "Traditional dances are still performed during village festivals.",
    "The traffic in Jakarta is very heavy during rush hour.",
    "Many people enjoy eating satay with peanut sauce.",
    "The rice fields look beautiful in the morning light.",
    "Although it was raining heavily, they still went to the market.",
    "The book that I borrowed from the library is very interesting.",
    "If you study hard, you will pass the exam easily.",
    "The restaurant where we had dinner last night was excellent.",
    "She has been working at the company for more than ten years.",
    "The project was completed successfully despite many challenges.",
    "I would like to visit my hometown during the next holiday.",
    "The meeting has been postponed until next Monday morning.",
    "He always tries to help his neighbors whenever possible.",
    "The movie we watched together was both funny and touching.",
]

INDONESIAN_TEXTS = [
    "Rubah cokelat yang gesit melompati anjing malas.",
    "Saya sangat suka makan makanan pedas di pasar malam.",
    "Dia pergi ke toko untuk membeli sayuran segar.",
    "Kucing itu sedang tidur di atas tikar.",
    "Dia minum kopi setiap pagi.",
    "Nenek saya membuat nasi goreng terbaik di desa.",
    "Kita akan bertemu di stasiun kereta besok pagi.",
    "Dia lupa membawa payung ketika hujan mulai turun.",
    "Adik saya membeli ponsel baru kemarin.",
    "Mereka sedang bermain sepak bola di lapangan.",
    "Saya biasanya bangun pukul enam pagi.",
    "Dia memasak makan malam untuk seluruh keluarga.",
    "Kemana kamu pergi akhir pekan lalu?",
    "Berapa harga baju ini?",
    "Bisakah kamu membantu saya dengan masalah ini?",
    "Jam berapa busnya tiba?",
    "Pemerintah mengumumkan peraturan baru untuk bisnis online minggu lalu.",
    "Saya harus menyelesaikan PR sebelum pergi ke bioskop bersama teman-teman.",
    "Kakek itu berjalan pelan di sepanjang sungai setiap sore.",
    "Banyak wisatawan mengunjungi Bali selama musim kemarau.",
    "Guru menjelaskan pelajaran dengan sangat jelas hari ini.",
    "Kami membeli buah dan sayuran segar dari pasar.",
    "Ayah saya bekerja sebagai guru di sekolah kecil.",
    "Anak-anak sangat senang bermain di taman.",
    "Presiden akan menyampaikan pidato di konferensi nasional.",
    "Beberapa perusahaan telah berinvestasi di proyek energi terbarukan.",
    "Undang-undang baru bertujuan melindungi konsumen dari penipuan.",
    "Ilmuwan menemukan spesies baru di hutan hujan.",
    "Ekonomi tumbuh lima persen tahun lalu.",
    "Saya menginstal aplikasi baru di laptop kemarin.",
    "Kecerdasan buatan sedang mengubah banyak industri dengan cepat.",
    "Mohon kirim laporan sebelum batas waktu besok.",
    "Koneksi internet sangat lambat hari ini.",
    "Dia bekerja secara remote dari kota kecil di Jawa.",
    "Kami merayakan Idul Fitri dengan mengunjungi kerabat dan makan bersama.",
    "Tarian tradisional masih ditampilkan saat festival desa.",
    "Lalu lintas di Jakarta sangat padat saat jam sibuk.",
    "Banyak orang menikmati sate dengan saus kacang.",
    "Sawah terlihat indah di pagi hari.",
    "Meskipun hujan deras, mereka tetap pergi ke pasar.",
    "Buku yang saya pinjam dari perpustakaan sangat menarik.",
    "Jika kamu belajar dengan giat, kamu akan lulus ujian dengan mudah.",
    "Restoran tempat kami makan malam kemarin sangat enak.",
    "Dia sudah bekerja di perusahaan itu selama lebih dari sepuluh tahun.",
    "Proyek itu berhasil diselesaikan meskipun banyak tantangan.",
    "Saya ingin mengunjungi kampung halaman saat liburan berikutnya.",
    "Rapat ditunda hingga Senin pagi minggu depan.",
    "Dia selalu berusaha membantu tetangganya kapan pun bisa.",
    "Film yang kami tonton bersama lucu sekaligus menyentuh.",
]

CHINESE_TEXTS = [
    "敏捷的棕色狐狸跳过了懒狗。",
    "我非常喜欢在夜市吃辣的食物。",
    "她去商店买了一些新鲜的蔬菜。",
    "猫在垫子上睡觉。",
    "他每天早上喝咖啡。",
    "我奶奶在村子里做最好的炒饭。",
    "我们明天早上在火车站见面。",
    "他下雨时忘记带雨伞了。",
    "我妹妹昨天买了一部新手机。",
    "他们在球场上踢足球。",
    "我通常早上六点起床。",
    "她为全家人做了晚饭。",
    "你上周末去哪里了？",
    "这件衬衫多少钱？",
    "你能帮我解决这个问题吗？",
    "公共汽车什么时候到？",
    "政府上周宣布了针对在线企业的新规定。",
    "我需要在和朋友去看电影之前完成作业。",
    "老人在河边慢慢地走着。",
    "许多游客在旱季访问巴厘岛。",
    "老师今天把课讲得很清楚。",
    "我们从市场买了一些新鲜的水果和蔬菜。",
    "我父亲在一所小学校当老师。",
    "孩子们在公园里玩得很开心。",
    "总统将在全国会议上发表演讲。",
    "几家公司已经投资了可再生能源项目。",
    "新法律旨在保护消费者免受欺诈。",
    "科学家在雨林中发现了一个新物种。",
    "去年经济增长了百分之五。",
    "我昨天在笔记本电脑上安装了一个新应用程序。",
    "人工智能正在迅速改变许多行业。",
    "请在明天截止日期前发送报告。",
    "今天的互联网连接非常慢。",
    "她在爪哇的一个小镇上远程工作。",
    "我们通过探亲和一起吃饭来庆祝开斋节。",
    "传统舞蹈在村节期间仍然在表演。",
    "雅加达在高峰时段交通非常拥堵。",
    "很多人喜欢吃花生酱沙爹。",
    "稻田在早晨的阳光下看起来很美。",
    "尽管雨下得很大，他们还是去了市场。",
    "我从图书馆借的那本书非常有趣。",
    "如果你努力学习，你会很容易通过考试。",
    "我们昨晚吃饭的餐厅非常好。",
    "她在那家公司工作了十多年。",
    "尽管有很多挑战，项目还是成功完成了。",
    "我下次假期想回老家看看。",
    "会议已推迟到下周一上午。",
    "他总是尽可能帮助邻居。",
    "我们一起看的电影既有趣又感人。",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="gpt2", help="HF tokenizer name or path")
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.tokenizer}")
    hf_tok = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer = StandardTokenizer(hf_tok)

    en_metrics = compute_metrics(tokenizer, ENGLISH_TEXTS, "en")
    id_metrics = compute_metrics(tokenizer, INDONESIAN_TEXTS, "id")
    zh_metrics = compute_metrics(tokenizer, CHINESE_TEXTS, "zh")

    print("\n=== English ===")
    print(en_metrics)
    print("\n=== Indonesian ===")
    print(id_metrics)
    print("\n=== Chinese ===")
    print(zh_metrics)


if __name__ == "__main__":
    main()
