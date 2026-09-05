# Cohort flow & event summary

## Sample-size flow
1. Downloaded/merged NHANES respondents (all ages): **101,316**
2. Linked to LMF (any ELIGSTAT): 101,316
3. Age 30-79 at exam: **41,267**
4. Mortality-eligible (ELIGSTAT==1) -> FINAL N: **41,151**

## Event counts & follow-up availability BY CYCLE
(follow-up = PERMTH_EXM/12, years)

| cycle | N | CVD death | competing | censored | median FU (y) | max FU (y) |
|---|---|---|---|---|---|---|
| 1999-2000 | 3,584 | 381 | 836 | 2,367 | 19.5 | 20.8 |
| 2001-2002 | 3,846 | 304 | 749 | 2,793 | 17.7 | 19.1 |
| 2003-2004 | 3,605 | 294 | 619 | 2,692 | 15.8 | 17.1 |
| 2005-2006 | 3,516 | 197 | 457 | 2,862 | 13.8 | 15.0 |
| 2007-2008 | 4,591 | 199 | 564 | 3,828 | 11.8 | 13.2 |
| 2009-2010 | 4,738 | 142 | 407 | 4,189 | 9.8 | 11.2 |
| 2011-2012 | 4,193 | 113 | 272 | 3,808 | 7.9 | 9.2 |
| 2013-2014 | 4,450 | 79 | 198 | 4,173 | 6.0 | 7.1 |
| 2015-2016 | 4,352 | 38 | 133 | 4,181 | 3.9 | 5.1 |
| 2017-2018 | 4,276 | 19 | 64 | 4,193 | 2.0 | 3.1 |
| **TOTAL** | 41,151 | 1766 | 4299 | 35,086 | | |

## Temporal-split guidance
- Cycles with **>=10y median follow-up** (good for long-horizon training/eval): 1999-2000, 2001-2002, 2003-2004, 2005-2006, 2007-2008
- Later cycles have shorter follow-up (mortality linkage through Dec 31 2019), so they carry fewer observed CVD deaths -> favour using earlier cycles for training and later cycles for temporal validation.
