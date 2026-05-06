# Dissertation Revision Prompt
# Use this file as the prompt for a new Claude chat to revise dissertation_collins_v4.docx
# Full prompt saved at: DISSERTATION_REVISION_PROMPT.md (see this file for the complete prompt)
# 
# QUICK SUMMARY OF CHANGES NEEDED:
# 1. No bare section titles - add bridge sentences before sub-headings
# 2. Sequential figure/table numbering (Fig 1-10, Table 1-7)  
# 3. Figure readability - min 12pt font on all diagrams
# 4. Add simulator screenshot placeholder as Figure 2 in Section 3.3
# 5. Expand simulator setup section (4 weeks of work)
# 6. Better section titles for 3.7-3.10
# 7. Rename 3.9 to Multi-Objective Optimisation, focus on top 2 params
# 8. Add AI model results (TabPFN-v2, TabNet, PatchTST, Chronos-2, AutoGluon)
# 9. RQ flow: results -> discussion -> finding -> next RQ
# 10. Define all abbreviations on first use
# 11. Expand conclusion: problem -> plan -> outcome -> reflection
# 12. Project document tone (first person where appropriate)
#
# AI MODEL RESULTS (actual pipeline output):
# TabPFN-v2:  launch_time=0.2104, brake_stopping=0.1843, step_yaw_rate=0.1802, launch_dist=0.1493
# Classical:  brake_stopping=0.222 (Stacking), launch_time=0.204 (RF), step_yaw_rate=0.149 (RF)
# TabPFN-v2 outperforms classical on 3/5 learnable KPIs
# Chronos-2 and PatchTST: all negative R2 (wrong model class for tabular regression)
# TabNet: numerical explosion on circle_avg_lat_g (R2 = -180)
# AutoGluon: partial results (5 of 13 targets NaN due to Windows cross-drive path issue)
#
# See dissertation_draft.md for the current full text baseline.
# The complete revision prompt with full dissertation text is too large for a single file -
# paste the content from the Claude Code session context into a new Claude chat.
