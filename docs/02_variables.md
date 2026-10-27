# 2. Variables

Descripciones copiadas de `data/raw/LCDataDictionary.xlsx`, hoja *LoanStats* (diccionario oficial incluido en el conjunto de Kaggle). Archivo generado por `src/preparar_datos.py`; no editar a mano.

| Variable | Uso | Descripción (diccionario) |
|---|---|---|
| `loan_amnt` | entrada (numérica) | The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value. |
| `int_rate` | entrada (numérica) | Interest Rate on the loan |
| `installment` | entrada (numérica) | The monthly payment owed by the borrower if the loan originates. |
| `annual_inc` | entrada (numérica) | The self-reported annual income provided by the borrower during registration. |
| `dti` | entrada (numérica) | A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income. |
| `delinq_2yrs` | entrada (numérica) | The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years |
| `inq_last_6mths` | entrada (numérica) | The number of inquiries in past 6 months (excluding auto and mortgage inquiries) |
| `mths_since_last_delinq` | entrada (numérica) | The number of months since the borrower's last delinquency. |
| `mths_since_last_record` | entrada (numérica) | The number of months since the last public record. |
| `mths_since_last_major_derog` | entrada (numérica) | Months since most recent 90-day or worse rating |
| `open_acc` | entrada (numérica) | The number of open credit lines in the borrower's credit file. |
| `pub_rec` | entrada (numérica) | Number of derogatory public records |
| `revol_bal` | entrada (numérica) | Total credit revolving balance |
| `revol_util` | entrada (numérica) | Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit. |
| `total_acc` | entrada (numérica) | The total number of credit lines currently in the borrower's credit file |
| `collections_12_mths_ex_med` | entrada (numérica) | Number of collections in 12 months excluding medical collections |
| `acc_now_delinq` | entrada (numérica) | The number of accounts on which the borrower is now delinquent. |
| `tot_coll_amt` | entrada (numérica) | Total collection amounts ever owed |
| `tot_cur_bal` | entrada (numérica) | Total current balance of all accounts |
| `total_rev_hi_lim` | entrada (numérica) | Total revolving high credit/credit limit |
| `antiguedad_credito` | entrada (numérica) | — (derivada: años entre earliest_cr_line e issue_d) |
| `term` | entrada (categórica) | The number of payments on the loan. Values are in months and can be either 36 or 60. |
| `grade` | entrada (categórica) | LC assigned loan grade |
| `sub_grade` | entrada (categórica) | LC assigned loan subgrade |
| `emp_length` | entrada (categórica) | Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.  |
| `home_ownership` | entrada (categórica) | The home ownership status provided by the borrower during registration. Our values are: RENT, OWN, MORTGAGE, OTHER. |
| `verification_status` | entrada (categórica) | Indicates if income was verified by LC, not verified, or if the income source was verified *(en el diccionario aparece como `is_inc_v`)* |
| `purpose` | entrada (categórica) | A category provided by the borrower for the loan request.  |
| `addr_state` | entrada (categórica) | The state provided by the borrower in the loan application |
| `initial_list_status` | entrada (categórica) | The initial listing status of the loan. Possible values are – W, F |
| `application_type` | entrada (categórica) | Indicates whether the loan is an individual application or a joint application with two co-borrowers |
| `funded_amnt` | solo LGD/EAD (posterior al desembolso) | The total amount committed to that loan at that point in time. |
| `funded_amnt_inv` | solo LGD/EAD (posterior al desembolso) | The total amount committed by investors for that loan at that point in time. |
| `out_prncp` | solo LGD/EAD (posterior al desembolso) | Remaining outstanding principal for total amount funded |
| `out_prncp_inv` | solo LGD/EAD (posterior al desembolso) | Remaining outstanding principal for portion of total amount funded by investors |
| `total_pymnt` | solo LGD/EAD (posterior al desembolso) | Payments received to date for total amount funded |
| `total_pymnt_inv` | solo LGD/EAD (posterior al desembolso) | Payments received to date for portion of total amount funded by investors |
| `total_rec_prncp` | solo LGD/EAD (posterior al desembolso) | Principal received to date |
| `total_rec_int` | solo LGD/EAD (posterior al desembolso) | Interest received to date |
| `total_rec_late_fee` | solo LGD/EAD (posterior al desembolso) | Late fees received to date |
| `recoveries` | solo LGD/EAD (posterior al desembolso) | post charge off gross recovery |
| `collection_recovery_fee` | solo LGD/EAD (posterior al desembolso) | post charge off collection fee |
| `last_pymnt_d` | solo LGD/EAD (posterior al desembolso) | Last month payment was received |
| `last_pymnt_amnt` | solo LGD/EAD (posterior al desembolso) | Last total payment amount received |
| `id` | identificador (se guarda para unir tablas, no entra al modelo) | A unique LC assigned ID for the loan listing. |
| `member_id` | excluida | A unique LC assigned Id for the borrower member. |
| `emp_title` | excluida | The job title supplied by the Borrower when applying for the loan.* |
| `issue_d` | fecha de originación: se usa para la partición temporal, no como entrada | The month which the loan was funded |
| `loan_status` | se usa para construir la variable objetivo `malo` | Current status of the loan |
| `pymnt_plan` | excluida | Indicates if a payment plan has been put in place for the loan |
| `url` | excluida | URL for the LC page with listing data. |
| `desc` | excluida | Loan description provided by the borrower |
| `title` | excluida | The loan title provided by the borrower |
| `zip_code` | excluida | The first 3 numbers of the zip code provided by the borrower in the loan application. |
| `earliest_cr_line` | excluida | The month the borrower's earliest reported credit line was opened |
| `next_pymnt_d` | excluida | Next scheduled payment date |
| `last_credit_pull_d` | excluida | The most recent month LC pulled credit for this loan |
| `policy_code` | excluida | publicly available policy_code=1
new products not publicly available policy_code=2 |
| `annual_inc_joint` | excluida | The combined self-reported annual income provided by the co-borrowers during registration |
| `dti_joint` | excluida | A ratio calculated using the co-borrowers' total monthly payments on the total debt obligations, excluding mortgages and the requested LC loan, divided by the co-borrowers' combined self-reported monthly income |
| `verification_status_joint` | excluida | Indicates if the co-borrowers' joint income was verified by LC, not verified, or if the income source was verified *(en el diccionario aparece como `verified_status_joint`)* |
| `open_acc_6m` | excluida | Number of open trades in last 6 months |
| `open_il_6m` | excluida | Number of currently active installment trades |
| `open_il_12m` | excluida | Number of installment accounts opened in past 12 months |
| `open_il_24m` | excluida | Number of installment accounts opened in past 24 months |
| `mths_since_rcnt_il` | excluida | Months since most recent installment accounts opened |
| `total_bal_il` | excluida | Total current balance of all installment accounts |
| `il_util` | excluida | Ratio of total current balance to high credit/credit limit on all install acct |
| `open_rv_12m` | excluida | Number of revolving trades opened in past 12 months |
| `open_rv_24m` | excluida | Number of revolving trades opened in past 24 months |
| `max_bal_bc` | excluida | Maximum current balance owed on all revolving accounts |
| `all_util` | excluida | Balance to credit limit on all trades |
| `inq_fi` | excluida | Number of personal finance inquiries |
| `total_cu_tl` | excluida | Number of finance trades |
| `inq_last_12m` | excluida | Number of credit inquiries in past 12 months |
