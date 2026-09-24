# CIK near-miss review sheet — 2026-09-24

Ticker intervals whose Wikipedia name has no exact SEC name match. Close matches were proposed by the named rules and kept only if the filer filed a 10-K or 10-Q during the interval. Decisions are recorded separately, in `cik_nearmiss_decisions_2026-09-24.json`, and only those are applied.

| # | ticker | interval | Wikipedia name | suggestion | basis |
|---|---|---|---|---|---|
| 1 | BCR | 1990-01-01 → 2018-01-03 | CR Bard | 9892 BARD C R INC /NJ/ | only close match that filed during the interval (same_letters: ['BARD C R INC /NJ/']) |
| 2 | BEAM | 1990-01-01 → 2014-05-01 | Suntory Global Spirits | 789073 Beam Suntory Inc. | WEAK (ticker-as-name only); confirm: only close match that filed during the interval (ticker_name: ['BEAM INC', 'FORTUNE BRANDS INC']) |
| 3 | BID | 1990-01-01 → 2019-10-03 | Sotheby's | 823094 SOTHEBYS | only close match that filed during the interval (same_letters: ['SOTHEBYS']) |
| 4 | BRCD | 1990-01-01 → 2017-11-22 | Brocade Communications | 1009626 BROCADE COMMUNICATIONS SYSTEMS INC | only close match that filed during the interval (all_words: ['BROCADE COMMUNICATIONS SYSTEMS INC']) |
| 5 | CA | 1990-01-01 → 2018-11-06 | CA Technologies | 356028 CA, INC. | only close match that filed during the interval (first_word: ['CA, INC.']) |
| 6 | CAA | 1990-01-01 → 2018-02-13 | CalAtlantic Homes | 878560 CalAtlantic Group, LLC | only close match that filed during the interval (first_word: ['CalAtlantic Group, Inc.', 'STANDARD PACIFIC CORP /DE/']) |
| 7 | CAB | 2012-10-04 → 2017-09-26 | Cabela's | 1267130 CABELAS INC | the only candidate matching by same_letters under a name held during the interval (['CABELAS INC']) |
| 8 | CMCSK | 2015-09-18 → 2015-12-15 | Comcast Class K Special, Comcast K Corp | 1166691 COMCAST CORP | only close match that filed during the interval (first_word: ['COMCAST CORP']) |
| 9 | DFS | 2007-07-02 → 2025-05-19 | Discover Financial | 1393612 Discover Financial Services | only close match that filed during the interval (all_words: ['Discover Financial Services']) |
| 10 | DWDP | 2017-09-01 → 2019-06-03 | DuPont | **needs judgment** | no close match that filed during the interval under a matching name; needs judgment |
| 11 | HAR | 1990-01-01 → 2017-03-16 | Harman International | 800459 HARMAN INTERNATIONAL INDUSTRIES INC /DE/ | only close match that filed during the interval (all_words: ['HARMAN INTERNATIONAL INDUSTRIES INC /DE/']) |
| 12 | HNZ | 1990-01-01 → 2013-06-06 | Heinz | 46640 Kraft Heinz Foods Co | only close match that filed during the interval (all_words: ['HEINZ H J CO']) |
| 13 | HOT | 1990-01-01 → 2016-09-22 | Starwood | 316206 STARWOOD HOTELS & RESORTS WORLDWIDE, LLC | only close match that filed during the interval (all_words: ['STARWOOD HOTEL & RESORTS WORLDWIDE, INC', 'STARWOOD HOTEL & RESORTS WORLDWIDE INC']) |
| 14 | HRC | 1990-01-01 → 2021-12-20 | Hillrom | 47518 Hill-Rom Holdings, Inc. | only close match that filed during the interval (same_letters: ['Hill-Rom Holdings, Inc.']) |
| 15 | IPG | 1990-01-01 → 2025-11-28 | Interpublic Group | 51644 INTERPUBLIC GROUP OF COMPANIES, INC. | only close match that filed during the interval (all_words: ['INTERPUBLIC GROUP OF COMPANIES, INC.']) |
| 16 | JCP | 1990-01-01 → 2013-12-02 | JCPenney | 1166126 Old COPPER Company, Inc. | only close match that filed during the interval (same_letters: ['J C PENNEY CO INC']) |
| 17 | MXIM | 2018-12-03 → 2021-08-30 | Maxim Integrated | 743316 MAXIM INTEGRATED PRODUCTS INC | only close match that filed during the interval (all_words: ['MAXIM INTEGRATED PRODUCTS INC']) |
| 18 | NDN | 1990-01-01 → 2012-01-13 | 99¢ Only Stores | 1011290 99 CENTS ONLY STORES LLC | only close match that filed during the interval (all_words: ['99 CENTS ONLY STORES']) |
| 19 | PRXL | 1990-01-01 → 2017-10-02 | PAREXEL | 799729 PAREXEL INTERNATIONAL CORP | only close match that filed during the interval (all_words: ['PAREXEL INTERNATIONAL CORP']) |
| 20 | RRD | 1990-01-01 → 2016-09-27 | RR Donnelley | 29669 RR Donnelley & Sons Co | only close match that filed during the interval (all_words: ['RR Donnelley & Sons Co']) |
| 21 | SKX | 1990-01-01 → 2025-09-12 | Skechers | 1065837 SKECHERS USA INC | only close match that filed during the interval (all_words: ['SKECHERS USA INC']) |
| 22 | SNV | 1990-01-01 → 2026-01-02 | Synovus | 18349 SYNOVUS FINANCIAL CORP | only close match that filed during the interval (all_words: ['SYNOVUS FINANCIAL CORP']) |
| 23 | SRC | 2019-07-09 → 2024-01-23 | Spirit Realty | 1308606 SPIRIT REALTY CAPITAL, INC. | ticker fit: SRC in ['SPIRIT REALTY CAPITAL, INC.']; co-filing LLC/LP dropped: [1703181] |
| 24 | TMHC | 2020-01-31 → 2026-07-24 | Taylor Morrison | 1562476 Taylor Morrison Home Corp | the only candidate matching by all_words under a name held during the interval (['Taylor Morrison Home Corp']) |
| 25 | TSS | 1990-01-01 → 2019-09-23 | TSYS | 1320760 TSS, Inc. | WEAK (ticker-as-name only); confirm: only close match that filed during the interval (ticker_name: ['Fortress International Group, Inc.', 'TSS, Inc.']) |
| 26 | TW | 1990-01-01 → 2016-01-05 | Towers Watson | 1470215 Towers Watson & Co. | only close match that filed during the interval (same_letters: ['JUPITER SATURN HOLDING CO', 'Towers Watson & Co.']) |
| 27 | VSCO | 2021-08-04 → 2023-06-19 | Victoria's Secret | 1856437 Victoria's Secret & Co. | only close match that filed during the interval (same_letters: ["Victoria's Secret & Co."]) |
| 28 | WBS | 1990-01-01 → 2026-08-20 | Webster Bank | 801337 WEBSTER FINANCIAL CORP | only close match that filed during the interval (first_word: ['WEBSTER FINANCIAL CORP']) |
| 29 | WRI | 1990-01-01 → 2021-08-04 | Weingarten Realty | 828916 WEINGARTEN REALTY INVESTORS /TX/ | only close match that filed during the interval (all_words: ['WEINGARTEN REALTY INVESTORS /TX/']) |
| 30 | X | 1990-01-01 → 2025-06-24 | U.S. Steel | 1163302 UNITED STATES STEEL CORP | the only candidate matching by same_letters under a name held during the interval (['UNITED STATES STEEL CORP']) |

## Proposals checked

**1. BCR** (1990-01-01 → 2018-01-03; Wikipedia: CR Bard)
- 9892 'BARD C R INC /NJ/' — same_letters 1.0; accepted
- 275110 'BARD C R INC                                          (PRED)' — same_letters 1.0; no_periodic_filing_in_interval
- 27224 'C. R. DAVIS & COMPANY' — first_word 0.433; no_periodic_filing_in_interval
- 1338413 'CR PACIFICLAND LLC' — first_word 0.433; no_periodic_filing_in_interval
- 1345424 'C R PROJECT LLC' — first_word 0.433; no_periodic_filing_in_interval
- 1351276 'CR CERTIFICATION CORP' — first_word 0.433; no_periodic_filing_in_interval
- 1428217 'CR ACQUISITION CORP.' — first_word 0.433; no_periodic_filing_in_interval
- 1436322 'CR WATERMAN & CO LLC' — first_word 0.433; no_periodic_filing_in_interval
- 1488806 'CR EVERGREEN, LLC' — first_word 0.433; no_periodic_filing_in_interval
- 1528968 'C R COLLECTIONS, LLC' — first_word 0.433; no_periodic_filing_in_interval

**2. BEAM** (1990-01-01 → 2014-05-01; Wikipedia: Suntory Global Spirits)
- 789073 'BEAM INC' — ticker_name 0.6; accepted
- 1381311 'BEAM INC' — ticker_name 0.6; no_periodic_filing_in_interval
- 1897953 'BEAM INC. /DE/' — ticker_name 0.6; no_periodic_filing_in_interval
- 1083808 'SUNTORY WATER GROUP INC' — first_word 0.4; no_periodic_filing_in_interval
- 1587200 'SUNTORY BEVERAGE & FOOD LIMITED/ADR' — first_word 0.367; no_periodic_filing_in_interval

**3. BID** (1990-01-01 → 2019-10-03; Wikipedia: Sotheby's)
- 823094 'SOTHEBYS' — same_letters 1.0; accepted
- 1420935 'SOTHEBY'S INTERNATIONAL REALTY, INC.' — all_words 0.8; no_periodic_filing_in_interval
- 2017944 'SOTHEBY'S ARTFI FUNDING LLC' — all_words 0.8; no_periodic_filing_in_interval
- 1420425 'SOTHEBY'S INTERNATIONAL REALTY LICENSEE LLC' — all_words 0.75; no_periodic_filing_in_interval
- 1420426 'SOTHEBY'S INTERNATIONAL REALTY AFFILIATES LLC' — all_words 0.75; no_periodic_filing_in_interval
- 1420427 'SOTHEBY'S INTERNATIONAL REALTY REFERRAL COMPANY, LLC' — all_words 0.75; no_periodic_filing_in_interval
- 1705822 'BID CORP LIMITED/ADR' — ticker_name 0.6; no_periodic_filing_in_interval

**4. BRCD** (1990-01-01 → 2017-11-22; Wikipedia: Brocade Communications)
- 1009626 'BROCADE COMMUNICATIONS SYSTEMS INC' — all_words 0.85; accepted
- 1493573 'BROCADE COMMUNICATIONS SYSTEMS SKYPORT LLC' — all_words 0.8; no_periodic_filing_in_interval
- 1408148 'BROCADE FUND LP' — first_word 0.433; no_periodic_filing_in_interval

**5. CA** (1990-01-01 → 2018-11-06; Wikipedia: CA Technologies)
- 356028 'CA, INC.' — first_word 0.5; accepted
- 1337773 'CA HOLDING INC' — first_word 0.5; no_periodic_filing_in_interval
- 1433857 'CA HOLDING INC' — first_word 0.5; no_periodic_filing_in_interval
- 1664562 'CA/ASV SPRING, LLC' — first_word 0.5; no_periodic_filing_in_interval
- 13463 'C A BOTZUM & CO.' — first_word 0.433; no_periodic_filing_in_interval
- 1026792 'CA SHORT CO' — first_word 0.433; no_periodic_filing_in_interval
- 1314355 'CA NETWORKS, INC.' — first_word 0.433; no_periodic_filing_in_interval
- 1315744 'CA LEWISPORT, LLC' — first_word 0.433; no_periodic_filing_in_interval
- 1319461 'CA MUER CORP' — first_word 0.433; no_periodic_filing_in_interval
- 1335241 'CA INDOSUEZ (SWITZERLAND) SA (CAIS)' — first_word 0.433; no_periodic_filing_in_interval

**6. CAA** (1990-01-01 → 2018-02-13; Wikipedia: CalAtlantic Homes)
- 1130466 'CALATLANTIC HOMES OF ARIZONA, INC.' — all_words 0.8; no_periodic_filing_in_interval
- 1309607 'CALATLANTIC HOMES OF INDIANA, INC.' — all_words 0.8; no_periodic_filing_in_interval
- 1354494 'CALATLANTIC HOMES OF TEXAS, INC.' — all_words 0.8; no_periodic_filing_in_interval
- 1738506 'CALATLANTIC HOMES OF WASHINGTON, INC.' — all_words 0.8; no_periodic_filing_in_interval
- 1738558 'CALATLANTIC HOMES OF GEORGIA, INC.' — all_words 0.8; no_periodic_filing_in_interval
- 878560 'CALATLANTIC GROUP, INC.' — first_word 0.5; accepted
- 1738406 'CALATLANTIC TITLE, INC.' — first_word 0.433; no_periodic_filing_in_interval

**7. CAB** (2012-10-04 → 2017-09-26; Wikipedia: Cabela's)
- 1267130 'CABELAS INC' — same_letters 1.0; accepted
- 1592143 'CABELA'S MASTER CREDIT CARD TRUST' — all_words 0.7; accepted
- 1592145 'CABELA'S CREDIT CARD MASTER NOTE TRUST' — all_words 0.65; accepted
- 1292774 'CABELA JAMES W' — first_word 0.4; no_periodic_filing_in_interval
- 1292776 'CABELA RICHARD N' — first_word 0.4; no_periodic_filing_in_interval
- 1292777 'CABELA MARY A' — first_word 0.4; no_periodic_filing_in_interval

**8. CMCSK** (2015-09-18 → 2015-12-15; Wikipedia: Comcast Class K Special, Comcast K Corp)
- 22301 'COMCAST CORP' — first_word 0.5; no_periodic_filing_in_interval
- 1166691 'COMCAST CORP' — first_word 0.5; accepted
- 275605 'COMCAST JOIN HOLDINGS INC' — first_word 0.433; no_periodic_filing_in_interval
- 732718 'COMCAST MO GROUP INC' — first_word 0.433; no_periodic_filing_in_interval
- 925692 'COMCAST CABLE HOLDINGS LLC' — first_word 0.433; no_periodic_filing_in_interval
- 939345 'COMCAST UK HOLDINGS INC' — first_word 0.433; no_periodic_filing_in_interval
- 1000450 'COMCAST LCI HOLDINGS INC' — first_word 0.433; no_periodic_filing_in_interval
- 1041854 'COMCAST CELLULAR HOLDINGS INC' — first_word 0.433; no_periodic_filing_in_interval
- 1105716 'COMCAST ICG INC' — first_word 0.433; no_periodic_filing_in_interval
- 1253778 'COMCAST PROGRAMING HOLDINGS INC' — first_word 0.433; no_periodic_filing_in_interval

**9. DFS** (2007-07-02 → 2025-05-19; Wikipedia: Discover Financial)
- 1393612 'DISCOVER FINANCIAL SERVICES' — all_words 0.85; accepted
- 1975918 'DISCOVER FINANCIAL HEALTH IMPROVEMENT FUND, LLC' — all_words 0.75; no_periodic_filing_in_interval
- 1761929 'DISCOVER GROUP LLC' — first_word 0.5; no_periodic_filing_in_interval
- 815743 'DISCOVER CREDIT CORP' — first_word 0.433; no_periodic_filing_in_interval
- 894327 'DISCOVER BANK' — first_word 0.433; no_periodic_filing_in_interval
- 1189699 'DISCOVER CAPITAL HOLDINGS CORP' — first_word 0.433; no_periodic_filing_in_interval
- 1383825 'DISCOVER SCREENS INC' — first_word 0.433; name_not_held_during_interval
- 1431213 'DISCOVER ADVISORY CO' — first_word 0.433; no_periodic_filing_in_interval
- 1610985 'DISCOVER CAPITAL, LLC' — first_word 0.433; no_periodic_filing_in_interval
- 1640497 'DISCOVER FUTURE LTD' — first_word 0.433; no_periodic_filing_in_interval

**10. DWDP** (2017-09-01 → 2019-06-03; Wikipedia: DuPont)
- 800299 'DUPONT CANADA INC                                      /FI' — all_words 0.85; no_periodic_filing_in_interval
- 1012128 'DUPONT PHOTOMASKS INC' — all_words 0.85; no_periodic_filing_in_interval
- 1027593 'DUPONT SECURITIES GROUP, INC.' — all_words 0.85; no_periodic_filing_in_interval
- 1072444 'DUPONT ENERGY CO' — all_words 0.85; no_periodic_filing_in_interval
- 1218729 'DUPONT ERIC' — all_words 0.85; no_periodic_filing_in_interval
- 1321570 'DUPONT HOSPITAL LLC' — all_words 0.85; no_periodic_filing_in_interval
- 1553335 'DUPONT JAKOB' — all_words 0.85; no_periodic_filing_in_interval
- 1932912 'DUPONT MEADOWS, LLC' — all_words 0.85; no_periodic_filing_in_interval
- 2036968 'DUPONT 37 LLC' — all_words 0.85; no_periodic_filing_in_interval
- 807904 'DUPONT DIRECT FINANCIAL HOLDINGS INC' — all_words 0.8; no_periodic_filing_in_interval

**11. HAR** (1990-01-01 → 2017-03-16; Wikipedia: Harman International)
- 45623 'HARMAN INTERNATIONAL INDUSTRIES INC/NY' — all_words 0.85; no_periodic_filing_in_interval
- 800459 'HARMAN INTERNATIONAL INDUSTRIES INC /DE/' — all_words 0.85; accepted
- 1640446 'HARMAN FINANCE INTERNATIONAL, S.C.A.' — all_words 0.8; no_periodic_filing_in_interval
- 901091 'HARMAN SIDNEY' — first_word 0.433; no_periodic_filing_in_interval
- 920236 'HARMAN AUTOMOTIVE INC' — first_word 0.433; no_periodic_filing_in_interval
- 1277950 'HARMAN PAUL' — first_word 0.433; no_periodic_filing_in_interval
- 1343559 'HARMAN GINA' — first_word 0.433; no_periodic_filing_in_interval
- 1401962 'HARMAN SUSAN' — first_word 0.433; no_periodic_filing_in_interval
- 1432071 'HARMAN BRYN' — first_word 0.433; no_periodic_filing_in_interval
- 1643077 'HARMAN TODD' — first_word 0.433; no_periodic_filing_in_interval

**12. HNZ** (1990-01-01 → 2013-06-06; Wikipedia: Heinz)
- 46640 'HEINZ H J CO' — all_words 0.85; accepted
- 918397 'HEINZ TERESA' — all_words 0.85; no_periodic_filing_in_interval
- 1446585 'HEINZ ROGER' — all_words 0.85; no_periodic_filing_in_interval
- 1473252 'HEINZ JAMES' — all_words 0.85; no_periodic_filing_in_interval
- 1599585 'HEINZ CREDIT LLC' — all_words 0.85; no_periodic_filing_in_interval
- 1599598 'HEINZ TRANSATLANTIC HOLDING LLC' — all_words 0.85; no_periodic_filing_in_interval
- 1599600 'HEINZ INVESTMENT CO' — all_words 0.85; no_periodic_filing_in_interval
- 1599696 'HEINZ PURCHASING CO' — all_words 0.85; no_periodic_filing_in_interval
- 1599700 'HEINZ THAILAND LTD' — all_words 0.85; no_periodic_filing_in_interval
- 1599706 'HEINZ GP LLC' — all_words 0.85; no_periodic_filing_in_interval

**13. HOT** (1990-01-01 → 2016-09-22; Wikipedia: Starwood)
- 316206 'STARWOOD LODGING CORP' — all_words 0.85; accepted
- 917112 'STARWOOD CAPITAL GROUP LP /ADV' — all_words 0.85; no_periodic_filing_in_interval
- 923030 'STARWOOD INVESTMENTS LP' — all_words 0.85; no_periodic_filing_in_interval
- 1095651 'STARWOOD FINANCIAL INC' — all_words 0.85; name_not_held_during_interval
- 1173886 'STARWOOD RESERVATIONS CORP' — all_words 0.85; no_periodic_filing_in_interval
- 1553289 'STARWOOD LAHAINA LLC' — all_words 0.85; no_periodic_filing_in_interval
- 48595 'STARWOOD LODGING TRUST' — all_words 0.8; no_periodic_filing_in_interval
- 831972 'STARWOOD FINANCIAL TRUST' — all_words 0.8; no_periodic_filing_in_interval
- 929867 'STARWOOD MEZZANINE INVESTORS LP' — all_words 0.8; no_periodic_filing_in_interval
- 1173887 'STARWOOD VACATION EXCHANGE CO' — all_words 0.8; no_periodic_filing_in_interval

**14. HRC** (1990-01-01 → 2021-12-20; Wikipedia: Hillrom)
- 47518 'HILL-ROM HOLDINGS, INC.' — same_letters 1.0; accepted

**15. IPG** (1990-01-01 → 2025-11-28; Wikipedia: Interpublic Group)
- 51644 'INTERPUBLIC GROUP OF COMPANIES INC' — all_words 0.75; accepted
- 944048 'INTERPUBLIC GROUP OF COMPANIES INC /IPG/' — all_words 0.75; no_periodic_filing_in_interval
- 1306956 'IPG (US) INC.' — ticker_name 0.6; no_periodic_filing_in_interval
- 1306958 'IPG HOLDINGS LP' — ticker_name 0.6; no_periodic_filing_in_interval
- 1306959 'IPG (US) HOLDINGS INC.' — ticker_name 0.6; no_periodic_filing_in_interval

**16. JCP** (1990-01-01 → 2013-12-02; Wikipedia: JCPenney)
- 77182 'J C PENNEY CORP INC' — same_letters 1.0; no_periodic_filing_in_interval
- 1166126 'J C PENNEY CO INC' — same_letters 1.0; accepted

**17. MXIM** (2018-12-03 → 2021-08-30; Wikipedia: Maxim Integrated)
- 743316 'MAXIM INTEGRATED PRODUCTS INC' — all_words 0.85; accepted
- 910468 'MAXIM GROUP INC /' — first_word 0.5; no_periodic_filing_in_interval
- 1172614 'MAXIM GROUP LLC' — first_word 0.5; no_periodic_filing_in_interval
- 1013351 'MAXIM PHARMACEUTICALS INC' — first_word 0.433; no_periodic_filing_in_interval
- 1048965 'MAXIM RETAIL GROUP' — first_word 0.433; no_periodic_filing_in_interval
- 1122380 'MAXIM MORTGAGE CORP/' — first_word 0.433; no_periodic_filing_in_interval
- 1313720 'MAXIM RESOURCES INC' — first_word 0.433; no_periodic_filing_in_interval
- 1366562 'MAXIM ENERGY CORP' — first_word 0.433; no_periodic_filing_in_interval
- 1422220 'MAXIM TEP, INC' — first_word 0.433; no_periodic_filing_in_interval
- 1434223 'MAXIM TEP LTD' — first_word 0.433; no_periodic_filing_in_interval

**18. NDN** (1990-01-01 → 2012-01-13; Wikipedia: 99¢ Only Stores)
- 1011290 '99 CENTS ONLY STORES' — all_words 0.85; accepted
- 1553724 '99 CENTS ONLY STORES' — all_words 0.85; no_periodic_filing_in_interval
- 1553585 '99 CENTS ONLY STORES TEXAS, INC.' — all_words 0.8; no_periodic_filing_in_interval
- 1270889 '99 COMMISSARY LLC' — first_word 0.4; no_periodic_filing_in_interval
- 1270893 '99 RESTAURANTS LLC' — first_word 0.4; no_periodic_filing_in_interval
- 1270895 '99 WEST INC' — first_word 0.4; no_periodic_filing_in_interval
- 1632783 '99 TAXIS' — first_word 0.4; no_periodic_filing_in_interval
- 1664675 '99 ADMINISTRATION INC' — first_word 0.4; no_periodic_filing_in_interval
- 1855001 '99 PATRIOTS, LLC' — first_word 0.4; no_periodic_filing_in_interval
- 1856110 '99 COMMERCIAL LLC' — first_word 0.4; no_periodic_filing_in_interval

**19. PRXL** (1990-01-01 → 2017-10-02; Wikipedia: PAREXEL)
- 799729 'PAREXEL INTERNATIONAL CORP' — all_words 0.85; accepted
- 1892518 'PAREXEL INTERNATIONAL CORP' — all_words 0.85; no_periodic_filing_in_interval
- 1732608 'PAREXEL INVESTMENT HOLDINGS, L.P.' — all_words 0.75; no_periodic_filing_in_interval

**20. RRD** (1990-01-01 → 2016-09-27; Wikipedia: RR Donnelley)
- 29669 'DONNELLEY R R & SONS CO' — all_words 0.8; accepted
- 930661 'DONNELLEY RR & SONS CO/ FA' — all_words 0.8; no_periodic_filing_in_interval
- 1132647 'RR DONNELLEY FINANCIAL /NY/FA/' — all_words 0.8; no_periodic_filing_in_interval
- 898430 'DONNELLEY R R & SONS CO 05/FA' — all_words 0.7; no_periodic_filing_in_interval
- 950132 'DONNELLEY R R & SONS CO 04/FA' — all_words 0.7; no_periodic_filing_in_interval
- 1316913 'RRD, INC.' — ticker_name 0.6; no_periodic_filing_in_interval
- 1426412 'RR (L STREET) INC' — first_word 0.5; no_periodic_filing_in_interval
- 1426414 'RR (BOULEVARD) INC' — first_word 0.5; no_periodic_filing_in_interval
- 1426415 'RR (SUPERIOR) INC' — first_word 0.5; no_periodic_filing_in_interval
- 1426416 'RR (ESSEN) INC' — first_word 0.5; no_periodic_filing_in_interval

**21. SKX** (1990-01-01 → 2025-09-12; Wikipedia: Skechers)
- 1065837 'SKECHERS USA INC' — all_words 0.8; accepted

**22. SNV** (1990-01-01 → 2026-01-02; Wikipedia: Synovus)
- 18349 'SYNOVUS FINANCIAL CORP' — all_words 0.85; accepted
- 776021 'SYNOVUS SECURITIES, INC.' — all_words 0.85; no_periodic_filing_in_interval
- 945597 'SYNOVUS TRUST CO/GA' — all_words 0.85; no_periodic_filing_in_interval
- 949887 'SYNOVUS TRUST CO' — all_words 0.85; no_periodic_filing_in_interval
- 1077694 'SYNOVUS TRUST CO/FL' — all_words 0.85; no_periodic_filing_in_interval
- 1520194 'SYNOVUS SECURITIES, INC.' — all_words 0.85; no_periodic_filing_in_interval
- 1268781 'SYNOVUS INVESTMENT ADVISORS INC' — all_words 0.8; no_periodic_filing_in_interval
- 1259075 'SYNOVUS TRUST CO NA' — all_words 0.75; no_periodic_filing_in_interval

**23. SRC** (2019-07-09 → 2024-01-23; Wikipedia: Spirit Realty)
- 1277406 'SPIRIT REALTY CAPITAL, INC.' — all_words 0.85; no_periodic_filing_in_interval
- 1308606 'SPIRIT REALTY CAPITAL, INC.' — all_words 0.85; accepted
- 1703181 'SPIRIT REALTY, L.P.' — all_words 0.85; accepted
- 1462515 'SRC GROUP, INC.' — ticker_name 0.6; no_periodic_filing_in_interval
- 1556628 'SPIRIT, INC.' — first_word 0.5; no_periodic_filing_in_interval
- 926032 'SPIRIT PARTNERS LP                                      /BD' — first_word 0.433; no_periodic_filing_in_interval
- 1190541 'SPIRIT ENERGY CORP' — first_word 0.433; no_periodic_filing_in_interval
- 1349682 'SPIRIT AEROSYSTEMS HOLDINGS INC' — first_word 0.433; no_periodic_filing_in_interval
- 1355054 'SPIRIT SHOP INC' — first_word 0.433; no_periodic_filing_in_interval
- 1364885 'SPIRIT AEROSYSTEMS HOLDINGS, INC.' — first_word 0.433; accepted

**24. TMHC** (2020-01-31 → 2026-07-24; Wikipedia: Taylor Morrison)
- 1562476 'TAYLOR MORRISON HOME CORP' — all_words 0.85; accepted
- 1804592 'TAYLOR MORRISON COMMUNITIES, INC.' — all_words 0.85; no_periodic_filing_in_interval
- 1610666 'TAYLOR GROUP CORP' — first_word 0.5; no_periodic_filing_in_interval
- 96536 'TAYLOR DEVICES, INC.' — first_word 0.433; accepted
- 735086 'TAYLOR DEVICES INC                                      /TA' — first_word 0.433; no_periodic_filing_in_interval
- 801966 'TAYLOR SECURITIES, INC.' — first_word 0.433; no_periodic_filing_in_interval
- 934373 'TAYLOR INVESTMENT CORP /MN/' — first_word 0.433; no_periodic_filing_in_interval
- 1012852 'TAYLOR DONALD' — first_word 0.433; no_periodic_filing_in_interval
- 1016861 'TAYLOR CAROLYN' — first_word 0.433; no_periodic_filing_in_interval
- 1018418 'TAYLOR JEFFREY' — first_word 0.433; no_periodic_filing_in_interval

**25. TSS** (1990-01-01 → 2019-09-23; Wikipedia: TSYS)
- 848013 'TSS LTD' — ticker_name 0.6; no_periodic_filing_in_interval
- 1320760 'TSS, INC.' — ticker_name 0.6; accepted

**26. TW** (1990-01-01 → 2016-01-05; Wikipedia: Towers Watson)
- 1470215 'TOWERS WATSON & CO.' — same_letters 1.0; accepted
- 1103126 'TOWERS WATSON DELAWARE HOLDINGS INC.' — all_words 0.85; name_not_held_during_interval
- 1426385 'TOWERS WATSON CAPITAL MARKETS INC.' — all_words 0.8; no_periodic_filing_in_interval
- 1594150 'TOWERS WATSON HEDGE ADVANTAGE FUND' — all_words 0.75; no_periodic_filing_in_interval
- 1594162 'TOWERS WATSON DIVERSIFYING STRATEGIES FUND' — all_words 0.75; no_periodic_filing_in_interval
- 1661667 'TOWERS WATSON ALTERNATIVE CREDIT FUND' — all_words 0.75; no_periodic_filing_in_interval
- 1934210 'TOWERS WATSON FOCUSED HIGH YIELD MASTER FUND' — all_words 0.65; no_periodic_filing_in_interval
- 1021387 'TW INC' — ticker_name 0.6; no_periodic_filing_in_interval
- 1357820 'TW/TAE, INC.' — ticker_name 0.6; no_periodic_filing_in_interval
- 1416026 'TW HOLDINGS INC' — ticker_name 0.6; no_periodic_filing_in_interval

**27. VSCO** (2021-08-04 → 2023-06-19; Wikipedia: Victoria's Secret)
- 1856437 'VICTORIA'S SECRET & CO.' — same_letters 1.0; accepted
- 1476447 'VICTORIA'S SECRET STORES, LLC' — all_words 0.85; no_periodic_filing_in_interval
- 1476448 'VICTORIA'S SECRET STORES BRAND MANAGEMENT, INC.' — all_words 0.75; no_periodic_filing_in_interval
- 1476449 'VICTORIA'S SECRET DIRECT BRAND MANAGEMENT, LLC' — all_words 0.75; no_periodic_filing_in_interval
- 1041378 'VICTORIA HOLDINGS INC' — first_word 0.433; no_periodic_filing_in_interval
- 1453380 'VICTORIA SECRET 3-D PROSPECT' — first_word 0.433; no_periodic_filing_in_interval
- 2020019 'VICTORIA'S RETAIL PORTFOLIO 1, LLC' — first_word 0.433; no_periodic_filing_in_interval
- 103466 'VICTORIA BANKSHARES INC' — first_word 0.4; no_periodic_filing_in_interval
- 720479 'VICTORIA FINANCIAL CORP' — first_word 0.4; no_periodic_filing_in_interval
- 796812 'VICTORIA CREATIONS INC' — first_word 0.4; no_periodic_filing_in_interval

**28. WBS** (1990-01-01 → 2026-08-20; Wikipedia: Webster Bank)
- 1105837 'WEBSTER BANK, N. A.' — all_words 0.85; no_periodic_filing_in_interval
- 1146252 'WEBSTER BANK TREASURY GROUP                             /MSD' — all_words 0.85; no_periodic_filing_in_interval
- 801337 'WEBSTER FINANCIAL CORP' — first_word 0.433; accepted
- 1012653 'WEBSTER JEFFREY' — first_word 0.433; no_periodic_filing_in_interval
- 1238363 'WEBSTER DAVID' — first_word 0.433; no_periodic_filing_in_interval
- 1297411 'WEBSTER I LLC' — first_word 0.433; no_periodic_filing_in_interval
- 1450640 'WEBSTER JEFFREY' — first_word 0.433; no_periodic_filing_in_interval
- 1463383 'WEBSTER SEAN' — first_word 0.433; no_periodic_filing_in_interval
- 1478045 'WEBSTER ATLANTIC CO' — first_word 0.433; no_periodic_filing_in_interval
- 1495228 'WEBSTER KAY' — first_word 0.433; no_periodic_filing_in_interval

**29. WRI** (1990-01-01 → 2021-08-04; Wikipedia: Weingarten Realty)
- 828916 'WEINGARTEN REALTY INVESTORS /TX/' — all_words 0.85; accepted
- 1268677 'WEINGARTEN TIM' — first_word 0.433; no_periodic_filing_in_interval
- 1437761 'WEINGARTEN IAN' — first_word 0.433; no_periodic_filing_in_interval
- 1866222 'WEINGARTEN TOMER' — first_word 0.433; no_periodic_filing_in_interval
- 1133800 'WEINGARTEN ALLAN D' — first_word 0.4; no_periodic_filing_in_interval
- 1135337 'WEINGARTEN ROBERT N' — first_word 0.4; no_periodic_filing_in_interval
- 1210577 'WEINGARTEN ALAN H' — first_word 0.4; no_periodic_filing_in_interval
- 1342556 'WEINGARTEN JOSEPH S' — first_word 0.4; no_periodic_filing_in_interval
- 1343379 'WEINGARTEN & WINBURN LLC' — first_word 0.4; no_periodic_filing_in_interval
- 1474807 'WEINGARTEN MICHAEL S' — first_word 0.4; no_periodic_filing_in_interval

**30. X** (1990-01-01 → 2025-06-24; Wikipedia: U.S. Steel)
- 317155 'UNITED STATES STEEL CORP/PA' — same_letters 1.0; no_periodic_filing_in_interval
- 1146318 'UNITED STATES STEEL CORP /PA/                           /TA' — same_letters 1.0; no_periodic_filing_in_interval
- 1163302 'UNITED STATES STEEL CORP' — same_letters 1.0; accepted
- 877374 'UNITED STATES STEEL & CARNEGIE PENSION FUND             /ADV' — all_words 0.7; no_periodic_filing_in_interval
- 1053136 'UNITED STATES STEEL & CARNEGIE PENSION FD' — all_words 0.7; no_periodic_filing_in_interval
- 1587298 'X SHARES, LLC' — ticker_name 0.6; no_periodic_filing_in_interval
- 1659790 'X/SEED CAPITAL, LP' — ticker_name 0.6; no_periodic_filing_in_interval
- 1661660 'X-CO HOLDINGS, LP' — ticker_name 0.6; no_periodic_filing_in_interval
- 1798841 'X, LLC' — ticker_name 0.6; no_periodic_filing_in_interval
- 27093 'U.S. GOLD CORP.' — first_word 0.5; accepted
