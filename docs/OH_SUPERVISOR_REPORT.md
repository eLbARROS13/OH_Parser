\begin{center}
\Large\textbf{Occupational Health Toolkit – Statistical Report for Supervisor}\\[0.4em]
\normalsize February 1, 2026
\end{center}

\begin{tcolorbox}[colback=infobg, colframe=infoblue, title={\faIcon{info-circle} Purpose}, fonttitle=\bfseries]
This report documents the statistical workflow, model choices, transformations, assumption checks, multiple-comparison corrections, and results for hypotheses H1–H6. It is a rigorous methods + results + interpretation briefing (not a paper) intended to justify why each model and adjustment was chosen and to summarize outputs with diagnostics.
\end{tcolorbox}

---

# Study Design and Data Structure

\begin{tcolorbox}[colback=tipbg, colframe=tipgreen, title={\faIcon{database} Dataset Summary}, fonttitle=\bfseries]
\begin{itemize}
\item Population: 38 office workers (Front-Office vs Back-Office)
\item Repeated measures: daily observations per subject (\texttt{subject\_id})
\item Key outcomes: EMG trapezius activity, perceived workload, sitting behavior, postural sway, and self-report sitting (OSPAQ)
\item Modeling strategy: Linear Mixed Models (LMMs) for repeated measures; OLS for subject-level validation (H4)
\item Repeated-measures implication: within-subject correlation requires random effects to avoid inflated type-I error
\end{itemize}
\end{tcolorbox}

---

# Hypotheses and Models (Confirmatory vs Exploratory)

All confirmatory models use ML estimation for valid likelihood ratio tests (LRT). Day effects are categorical (\texttt{C(day\_index)}), avoiding linear-trend assumptions.

\clearpage
\begin{landscape}
\footnotesize
\setlength{\LTpre}{0pt}
\setlength{\LTpost}{0pt}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.2}
\setlength{\LTleft}{0pt}
\setlength{\LTright}{0pt}
\begin{longtable}{p{3.0cm}p{2.8cm}p{1.4cm}p{8.0cm}p{4.5cm}}
\textbf{Hypothesis} & \textbf{Outcome} & \textbf{Model} & \textbf{Formula (fixed effects)} & \textbf{Notes} \\
\hline
H1 (Confirmatory) & EMG p90 (\%MVC) & LMM & log(EMG p90) ~ \texttt{work\_type} + \texttt{C(day\_index)} & Log transform for skew/heteroscedasticity; EMG p90 not bounded in [0,1] \\
H2 (Confirmatory) & Workload mean & LMM & \texttt{workload\_mean} ~ \texttt{work\_type} + \texttt{C(day\_index)} & No transform \\
H3 (Confirmatory) & Sitting proportion & LMM & logit(\texttt{har\_sentado\_prop}) ~ \texttt{workload\_mean} + \texttt{work\_type} + \texttt{C(day\_index)} & Proportion -> logit \\
H4 (Confirmatory) & OSPAQ validation & OLS & logit(\texttt{har\_sentado\_prop}) ~ \texttt{ospaq\_sitting\_frac} + \texttt{work\_type} & Subject-level aggregation \\
H5 (Exploratory) & EMG p90 (\%MVC) & LMM & EMG p90 ~ \texttt{hr\_ratio\_mean\_within} + \texttt{hr\_ratio\_mean\_between} + \texttt{noise\_mean\_within} + \texttt{noise\_mean\_between} + \texttt{posture\_95\_confidence\_ellipse\_area\_within} + \texttt{posture\_95\_confidence\_ellipse\_area\_between} + \texttt{work\_type} + \texttt{C(day\_index)} & Within-between decomposition \\
H6 (Confirmatory) & Posture area ($cm^2$) & LMM & \texttt{posture\_95\_confidence\_ellipse\_area} ~ \texttt{work\_type} + \texttt{C(day\_index)} & No transform \\
\end{longtable}
\end{landscape}

---

# Transformations and Units

\begin{tcolorbox}[colback=lightgray, colframe=darkgray, title={\faIcon{sliders-h} Transform Summary}, fonttitle=\bfseries]
\begin{itemize}
\item EMG p90: \%MVC; can exceed 100\% -> not a strict proportion; log transform used in H1
\item Workload mean: questionnaire score; no transform
\item Sitting proportion: proportion in (0,1); logit transform used in H3 and H4
\item OSPAQ sitting: proportion in (0,1); predictor, no transform
\item Posture ellipse area: $cm^2$; no transform
\item HAR durations: seconds; used for duration-weighted sitting proportion in H4
\end{itemize}
\end{tcolorbox}

---

# Assumption Checks and Corrections

\begin{tcolorbox}[colback=warningbg, colframe=warningyellow, title={\faIcon{exclamation-triangle} Diagnostics}, fonttitle=\bfseries]
\begin{itemize}
\item Normality: Q–Q plot + Shapiro-Wilk/Jarque-Bera summary
\item Homoscedasticity: residuals vs fitted + Breusch–Pagan proxy
\item Outliers: standardized residuals > 3 flagged
\item Auto-correction: if violations and outcome >= 0, apply log transform and refit
\item Bootstrap: cluster bootstrap p-values if violations persist and configured
\end{itemize}
\end{tcolorbox}

Note: H5 produced convergence warnings (boundary of parameter space), retained to avoid masking instability in the exploratory model.

---

# Multiple Comparisons

Confirmatory family: H1, H2, H3, H4, H6

\begin{tcolorbox}[colback=infobg, colframe=infoblue, title={\faIcon{check-circle} Correction Strategy}, fonttitle=\bfseries]
Holm step-down procedure (FWER control) applied to the confirmatory family. H5 is exploratory and excluded from correction. Primary p-values are LRT (full vs reduced model); Wald p-values retained for sensitivity.
\end{tcolorbox}

---

# Results: Estimates + Interpretation

\clearpage
\begin{landscape}
\footnotesize
\setlength{\LTpre}{0pt}
\setlength{\LTpost}{0pt}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.2}
\setlength{\LTleft}{0pt}
\setlength{\LTright}{0pt}
\begin{longtable}{p{2.0cm}p{1.9cm}p{2.4cm}p{4.2cm}p{1.1cm}p{1.1cm}p{1.1cm}p{7.0cm}}
\textbf{Hypothesis} & \textbf{N\_obs / N\_subjects} & \textbf{Primary term} & \textbf{Estimate (95\% CI)} & \textbf{Wald p} & \textbf{LRT p} & \textbf{Holm p} & \textbf{Interpretation} \\
\hline
H1 & 161 / 38 & \texttt{work\_type} & 0.5027 [0.1335, 0.8719] & 0.0076 & 0.0107 & 0.0537 & FO higher than BO on log scale; narrowly misses Holm threshold \\
H2 & 176 / 38 & \texttt{work\_type} & -0.0176 [-0.3930, 0.3578] & 0.9268 & 0.9268 & 1.0000 & No evidence of FO/BO workload difference \\
H3 & 168 / 38 & \texttt{workload\_mean} & -0.0505 [-0.1416, 0.0405] & 0.2768 & 0.0431 & 0.1294 & LRT suggests model-level improvement but coefficient not significant; tentative evidence \\
H4 & 38 / 38 & \texttt{ospaq\_sitting\_frac} & 0.1709 [-0.4316, 0.7734] & 0.5684 & — & 1.0000 & No evidence of strong self-report vs objective association \\
H5 (Expl.) & 160 / 38 & \texttt{posture\_within} & 0.0041 [-0.0834, 0.0916] & 0.9266 & 0.9266 & — & No within-day posture–EMG association; exploratory with convergence warnings \\
H6 & 180 / 38 & \texttt{work\_type} & -0.1482 [-0.2660, -0.0305] & 0.0136 & 0.0174 & 0.0698 & Suggestive FO/BO difference; not confirmatory under Holm \\
\end{longtable}
\end{landscape}

---

# Plots and Diagnostics

\clearpage

Each LMM includes a 4-panel summary: trajectories, random intercepts, Q–Q plot, residuals vs fitted.

	extbf{H1 – EMG p90}\\
\begin{center}
\includegraphics[width=0.9\textwidth]{plots/hypotheses/H1\_summary.png}
\includegraphics[width=0.9\textwidth]{plots/hypotheses/H1\_group\_comparison.png}
\end{center}
\textit{Diagnostics: improved normality after log transform; residual variance stabilized.}

	extbf{H2 – Workload}\\
\begin{center}
\includegraphics[width=0.9\textwidth]{plots/hypotheses/H2\_summary.png}
\includegraphics[width=0.9\textwidth]{plots/hypotheses/H2\_group\_comparison.png}
\end{center}
\textit{Diagnostics: approximately symmetric residuals; no strong heteroscedasticity.}

	extbf{H3 – Workload → Sitting}\\
\begin{center}
\includegraphics[width=0.9\textwidth]{plots/hypotheses/H3\_summary.png}
\includegraphics[width=0.9\textwidth]{plots/hypotheses/H3\_workload\_vs\_sitting.png}
\end{center}
\textit{Diagnostics: logit scale yields acceptable residual structure; mild tail deviation.}

	extbf{H4 – OSPAQ Validation}\\
\begin{center}
\includegraphics[width=0.9\textwidth]{plots/hypotheses/H4\_ospaq\_vs\_objective.png}
\includegraphics[width=0.9\textwidth]{plots/hypotheses/H4\_ols\_diagnostics.png}
\end{center}
\textit{Diagnostics: OLS residuals show no strong pattern; normality acceptable for N=38.}

	extbf{H5 – Physiological → EMG (Exploratory)}\\
\begin{center}
\includegraphics[width=0.9\textwidth]{plots/hypotheses/H5\_summary.png}
\includegraphics[width=0.9\textwidth]{plots/hypotheses/H5\_posture\_vs\_emg.png}
\end{center}
\textit{Diagnostics: convergence warnings; interpret cautiously.}

	extbf{H6 – Posture}\\
\begin{center}
\includegraphics[width=0.9\textwidth]{plots/hypotheses/H6\_summary.png}
\includegraphics[width=0.9\textwidth]{plots/hypotheses/H6\_group\_comparison.png}
\end{center}
\textit{Diagnostics: residuals broadly acceptable.}

---

# Limitations and Practical Considerations

\begin{itemize}
\item Sample size (38 subjects) limits power under strict FWER control
\item LMMs assume Gaussian residuals on the transformed scale; diagnostics support adequacy but not certainty
\item EMG p90 can exceed 100\% MVC; logit is invalid for this outcome
\item H5 is exploratory and relatively complex for the available sample size
\item H4 is cross-sectional; Bland–Altman could be added if agreement (not only correlation) is desired
\end{itemize}

---

# Recommendations for Next Steps

\begin{itemize}
\item Provide an FDR sensitivity analysis if a less conservative correction is desired
\item Consider power analysis for future data collections
\item Add Bland–Altman analysis for OSPAQ validation if agreement metrics are needed
\end{itemize}

---

# Recommendations for Next Steps

\begin{itemize}
\item Provide an FDR sensitivity analysis if a less conservative correction is desired
\item Consider power analysis for future data collections
\item Add Bland–Altman analysis for OSPAQ validation if agreement metrics are needed
\end{itemize}
