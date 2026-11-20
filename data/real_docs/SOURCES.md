# Real Indian public-record documents

Public contract and tender documents issued by Indian government bodies, used to test
the tagger on text it was not trained on (it is trained on US SEC contracts, LEDGAR).
Downloaded 2026-11-20. Each is a public document from the URL below.

| File | What it is | Issued by | Pages | Source |
|---|---|---|---|---|
| `gem_gtc_v1_10.pdf` | General Terms and Conditions on GeM 3.0 (v1.10, 31 Jan 2019): the contract terms behind every Government e-Marketplace purchase | Government e-Marketplace (GeM), Ministry of Commerce | 33 | https://assets-bg.gem.gov.in/resources/pdf/GTC_on_GeM_v1.10_31Jan19.pdf |
| `ivri_gcc.pdf` | General Conditions of Contract used in the institute's tenders | ICAR-Indian Veterinary Research Institute | 20 | https://ivri.nic.in/tender/gcc.pdf |
| `imd_tender_td310.pdf` | A real tender (GEM/2025/B/5945744): procurement of 34,400 GPS radiosondes and ground systems, with full contract conditions | India Meteorological Department, Ministry of Earth Sciences | 152 | https://internal.imd.gov.in/tenders/20250224_td_310.pdf |
| `mtd_goods_bpcl_copy.pdf` *(not committed, 51 MB)* | Model Tender Document for Procurement of Goods (Department of Expenditure, Ministry of Finance): the template central ministries base goods tenders on. Copy hosted by Bharat Petroleum | Ministry of Finance | 210 | https://www.bharatpetroleum.in/images/files/Model%20Tender%20Document%20for%20Procurement%20of%20Goods.pdf |

The official copy of the Model Tender Document on eprocure.gov.in refused automated downloads,
so the Bharat Petroleum copy is used; its first pages are scanned images, the rest is text.
It is too large for the repository: download it from the URL above into this folder to reproduce
the full real-records evaluation (the extracted clauses and hand labels are committed either way).
