# AI 학습 데이터 보상 현황

*2026-10-25 기준. 공개된 사실만 담았고, 공개되지 않은 항목은 "비공개", 확인하지 못한 항목은 "미확인"으로 표시했다.*

---

## 1. 학습 데이터 보상 제도 (운영 중이거나 발표됨)

| 곳 | 분야 | 분배 기준 | 비율·금액 | 주기 | 동의 방식 |
|---|---|---|---|---|---|
| **Adobe** (Firefly 보너스) | 이미지·영상 | ① 학습 대상이 된 자산 수 ② 같은 12개월 동안의 라이선스 판매 수 | 비공개. Adobe 재량으로 정함 | 연 1회. 2026년이 4년째(9/17 지급) | opt-out 불가 |
| **Getty Images** | 이미지 | ① 학습 데이터셋에 들어간 자산 수에 비례 ② 기존 라이선스 매출에 비례 | 비공개 ("set formula") | 연 1회 | 미확인 |
| **Shutterstock** (Contributor Fund) | 이미지 | 데이터셋 판매 계약금을 기여자들과 나눔. 라이브러리 기여량에 비례 | 회사 발표 없음. 독립 설문(기여자 58명) 결과 이미지당 평균 $0.0078, 중앙값 $0.0069 | 연 2회 | 미확인 |
| **Bria** | 이미지 (음악으로 확장 중) | 생성물마다 attribution을 계산하고, 데이터 소유자별 영향도로 배분 | Bria와 데이터 소유자 측이 "equally" 나눔. 데이터 파트너 30곳 이상(Getty, Alamy, Freepik 등) | 미확인 | 계약한 파트너만 |
| **STIM** (스웨덴 음악 저작권 단체, 2025.9 시작) | 음악 작곡 (녹음은 제외) | 학습분: 라이선스에 포함된 모든 작품에 균등 배분. 출력분: attribution(Sureel) 기준. attribution을 신뢰하기 어려우면 학습 이용 기준으로 유추 | 요율 비공개 | 라이선시가 attribution 분석을 월 단위로 제출 | 작품 단위 opt-in. 2026.6.1부터 AI 권리 카테고리 운영 |
| **GEMA** (독일 음악 저작권 단체, 2024.9 발표) | 음악 | AI 사업자 순수익의 일정 비율 + 출력량에 연동한 최소 로열티 | 30% | 미확인 | 개별 작품 attribution 언급 없음 |
| **CCC** AI Systems Training License (미국 저작권 집중관리 기관) | 텍스트 등 | 비공개 | 비공개 | 미확인 | 자발적·비독점 집단 라이선스 |
| **음반사–AI 회사 계약** (WMG–Suno 2025.11, BMG–Suno 2026.8, UMG–Spotify 2026.5 등) | 음악 | 비공개 | 비공개 | 비공개 | WMG와 BMG는 아티스트 opt-in 방식 |

**공통으로 공개되지 않은 것:** 창작자 1인당 실제 지급액, 그리고 attribution 결과가 정확한지 검증한 수치.

---

## 2. 참고: 출력·크롤링 기준 보상 (학습 데이터가 아니라 답변·접근이 기준)

| 곳 | 분배 기준 | 비율 |
|---|---|---|
| **ProRata** (Gist) | AI 답변에 쓰인 콘텐츠 양에 비례 | 광고수익의 50% |
| **Perplexity** Comet Plus (2025.10) | 사람 방문 수, 검색 인용, 에이전트 행동 | 구독수익의 80% |
| **Cloudflare** Pay-per-crawl (2025.7) | 크롤 1회당 통행료. 돈을 내지 않는 봇은 기술적으로 차단 | 사이트별로 설정 |
| **RSL** (2025.9) | robots.txt에 라이선스 조건을 명시하는 방식. 로열티나 출처 표시를 요구할 수 있음 | 강제력 없음 |

---

## 3. 한국 현황

- **라이선스·보상 체계:** 없음. 한음저협은 AI 수익을 배분하려면 "라이선스 체계가 우선 마련돼야" 한다는 입장이다. 배분 방식과 기준도 아직 정해지지 않았다 (2026.9).
- **한음저협 AI 음악 등록 기준:** 2026.7.28 의결 → 8.3 시행 → 22일 만에 철회. 국회·문체부가 추가 논의를 요청했다. 기사에 따르면 AI 사용 여부와 비율을 검증할 방법이 없어 자율 신고에 의존하던 상태였다.
- **권리자 6개 단체 연합** (2026.2.26 출범)
  - 요구: 동의 없는 학습 금지, 생성 과정 투명성, 인간 창작물과 AI 결과물의 구분
  - 계획: ISWC·ISRC·YouTube CID·UCI 코드를 블록체인으로 연동해 이용 내역을 실시간으로 추적·분배하는 인프라. 공동 기금과 단일 협상 창구도 만든다.
  - 음콘협 회원사: 하이브, SM, JYP, YG, 카카오엔터, Sony, Warner, Universal
- **정부**
  - AI 행동계획(안) 32번 (2025.12): 권리자가 불명확한 저작물에 한해 "선사용 후보상". opt-out은 존중한다.
  - 문체부 공정이용 안내서 (2026.2): 거래시장이 있는 분야는 거래를 지원하고 opt-out을 활성화한다. 수익 공유 체계는 "향후 마련".
  - 국가AI전략위: AI 학습용 표준계약서를 고시하도록 권고했다.
- **국내 기업: 뉴튠(Neutune)**
  - 생성할 때 실제로 쓰인 음악 블록(8~16마디 단위)을 기록해 정산과 연결한다.
  - Music DNA로 유사도를 탐지한다.
  - DDEX에 국내 첫 회원으로 가입했다.
  - 학습 데이터 기반 attribution은 아니다.

---

## Sources
- [Adobe: Firefly FAQ for Stock Contributors](https://helpx.adobe.com/lt/stock/contributor/help/firefly-faq-for-adobe-stock-contributors.html)
- [TechCrunch: How much can artists make from generative AI (2023)](https://techcrunch.com/2023/09/30/how-much-can-artists-make-from-generative-ai-vendors-wont-say/)
- [Digiday: Bria models trained on Getty Images](https://digiday.com/media-buying/startup-debuts-new-ai-models-trained-on-getty-images-and-other-content-as-copyright-concerns-loom/)
- [MBW: Bria raises $40M](https://www.musicbusinessworldwide.com/bria-raises-40m-to-expand-patented-attribution-engine-to-ai-generated-music-content)
- [STIM: world's first AI license for music](https://www.stim.se/en/news/stim-launches-the-worlds-first-ai-license-for-music)
- [Interspace: STIM AI rights category (2026-10-08)](https://interspacemusic.com/blog/stim-ai-rights-category-ai-licence-writer-approval/)
- [GEMA: generative AI licensing model](https://www.gema.de/en/w/generative-ai-licensing-model)
- [CCC: AI Systems Training License overview](https://www.copyright.com/wp-content/uploads/2026/03/AISTL-Overview-Sheet.pdf)
- [딜사이트: 음원사 라이선스 전환·국내 동향 (2026-09-11)](https://dealsite.co.kr/articles/168799)
- [SiliconANGLE: ProRata raises $40M](https://siliconangle.com/2025/09/05/prorata-raises-40m-develop-ai-tools-publishers/)
- [Press Gazette: Perplexity Comet Plus](https://pressgazette.co.uk/news/perplexity-launches-ai-subscription-revenue-share-scheme-for-publishers/)
- [Plagiarism Today: RSL vs Pay-Per-Crawl](https://www.plagiarismtoday.com/2025/09/11/ai-licensing-comparison-rsl-vs-pay-per-carwl/)
- [한국경제: 음저협 AI 등록 기준 철회 (2026-10-03)](https://www.hankyung.com/article/202610020512H)
- [MBW: Korea's music rights groups form coalition (2026-03-04)](https://www.musicbusinessworldwide.com/south-koreas-music-rights-groups-form-coalition-to-confront-ai-copyright-crisis)
- [머니투데이: '선 사용 후 보상' (2026-01-15)](https://www.mt.co.kr/tech/2026/01/15/2026011513430579722)
- [머니투데이: 정부 K-콘텐츠 AI 생태계 (2026-03-11)](https://www.mt.co.kr/tech/2026/03/11/2026022710581198026)
- [헬로티: 뉴튠 인터뷰 (2026-03-20)](https://www.hellot.net/mobile/article.html?no=111512)