setwd("V:/evenmmV")
library(jsonlite)
color_palette = c("#c19d13", "#1765ac")
#list_names = c("pfs_times", "pfs_censoreds", "pfs_eventdescs", "Treat_is_Kd", "rho_r", "rho_s", "pi_r")
pfs_times = fromJSON("pfs_times.json") / 30.4325 #months
pfs_censoreds = fromJSON("pfs_censoreds.json")
pfs_eventdescs = fromJSON("pfs_eventdescs.json")
Treat_is_Kd = fromJSON("Treat_is_Kd.json")
rho_r = fromJSON("rho_r.json")
rho_s = fromJSON("rho_s.json")
pi_r = fromJSON("pi_r.json")

pfs_events = 1 - pfs_censoreds

median_rho_s = median(rho_s)
median_rho_r = median(rho_r)
median_pi_r = median(pi_r)

hi_rho_s = ifelse(rho_s > median(rho_s), 1, 0)
hi_rho_r = ifelse(rho_r > median(rho_r), 1, 0)
hi_pi_r = ifelse(pi_r > median(pi_r), 1, 0)

data = data.frame(
  pfs_times = pfs_times, 
  pfs_censoreds = pfs_censoreds,
  Treat_is_Kd = Treat_is_Kd,
  rho_s = rho_s,
  rho_r = rho_r, 
  pi_r = pi_r,
  hi_rho_s = hi_rho_s,
  hi_rho_r = hi_rho_r, 
  hi_pi_r = hi_pi_r
)

library(survival)
surv_object = with(data, Surv(pfs_times, pfs_events))

# for ggsurvplot and p value
#library(survminer)

# plain PFS Survival
surv_plain = survfit(surv_object ~ 1)
plot(surv_plain, main = "Survival curve", xlab = "pfs_times", ylab = "PFS (%)")

surv_treat_is_Kd = survfit(surv_object ~ Treat_is_Kd)
plot(surv_treat_is_Kd, main = "Survival curve", xlab = "pfs_times", ylab = "PFS (%)")

# Regress on treatment 
cox_model = coxph(Surv(pfs_times, pfs_events) ~ Treat_is_Kd)
summary(cox_model)
surv_rho_s = survfit(surv_object ~ Treat_is_Kd)
plot(surv_treat_is_Kd, main = "Survival curve", xlab = "pfs_times", ylab = "PFS (%)")
survdiff(Surv(pfs_times, pfs_events) ~ Treat_is_Kd, data=data)

# rho s
surv_fit_rho_s = survfit(surv_object ~ strata(hi_rho_s), data=data)
# no CI, with text and save as pdf
pdf_file_rho_s = "surv_rho_s.pdf"
pdf(pdf_file_rho_s, width=7,height=5)
plot(surv_fit_rho_s, main = "Survival curve", xlab = "pfs_times", ylab = "PFS (%)", col=c(1,2), conf.int = FALSE)
legend("bottomleft", legend = c("rho_s below median", "rho_s above median"), col=c(1,2), lty=1:1)
cox_rho_s = coxph(Surv(pfs_times, pfs_events) ~ hi_rho_s)
summary_cox_rho_s = summary(cox_rho_s)
pvalue_rho_s = summary_cox_rho_s$coefficients[1, "Pr(>|z|)"]
hazard_ratio_rho_s = summary_cox_rho_s$coefficients[1, "exp(coef)"]
ci_low_rho_s = summary_cox_rho_s$conf.int[1, "lower .95"]
ci_high_rho_s = summary_cox_rho_s$conf.int[1, "upper .95"]
pvalue_text_rho_s = ifelse(pvalue_rho_s < 0.001, "p-value < 0.001", paste("p-value: ", round(pvalue_rho_s, 4)))
text_info_rho_s = paste(pvalue_text_rho_s, 
                  "\nHR: ", round(hazard_ratio_rho_s, 2), 
                  "\n95% CI: (", round(ci_low_rho_s, 2), " - ", round(ci_high_rho_s, 2), ")", sep="")
legend(-1, 0.5, legend = text_info_rho_s, bg="transparent", bty="n")
dev.off()

# with CI plotted
pdf_file_rho_s = "surv_rho_s_CI.pdf"
pdf(pdf_file_rho_s, width=7,height=5)
plot(surv_fit_rho_s, main = "Survival curve", xlab = "pfs_times", ylab = "PFS (%)", col=c(1,2), conf.int = TRUE)
legend("bottomleft", legend = c("rho_s below median", "rho_s above median"), col=c(1,2), lty=1:1)
legend(-1, 0.5, legend = text_info_rho_s, bg="transparent", bty="n")
dev.off()

survdiff(Surv(pfs_times, pfs_events) ~ hi_rho_s, data=data)

# rho r
surv_fit_rho_r = survfit(surv_object ~ strata(hi_rho_r))
# no CI, with text and save as pdf
pdf_file_rho_r = "surv_rho_r.pdf"
pdf(pdf_file_rho_r, width=7,height=5)
plot(surv_fit_rho_r, main = "Survival curve", xlab = "pfs_times", ylab = "PFS (%)", col=c(1,2), conf.int = FALSE)
legend("bottomleft", legend = c("rho_r below median", "rho_r above median"), col=c(1,2), lty=1:1)

cox_rho_r = coxph(Surv(pfs_times, pfs_events) ~ hi_rho_r)
summary_cox_rho_r = summary(cox_rho_r)
pvalue_rho_r = summary_cox_rho_r$coefficients[1, "Pr(>|z|)"]
hazard_ratio_rho_r = summary_cox_rho_r$coefficients[1, "exp(coef)"]
ci_low_rho_r = summary_cox_rho_r$conf.int[1, "lower .95"]
ci_high_rho_r = summary_cox_rho_r$conf.int[1, "upper .95"]
pvalue_text_rho_r = ifelse(pvalue_rho_r < 0.001, "p-value < 0.001", paste("p-value: ", round(pvalue_rho_r, 4)))
text_info_rho_r = paste(pvalue_text_rho_r, 
                       "\nHR: ", round(hazard_ratio_rho_r, 2), 
                        "\n95% CI: (", round(ci_low_rho_r, 2), " - ", round(ci_high_rho_r, 2), ")", sep="")
legend(-1, 0.5, legend = text_info_rho_r, bg="transparent", bty="n")
dev.off()

# with CI plotted
pdf_file_rho_r = "surv_rho_r_CI.pdf"
pdf(pdf_file_rho_r, width=7,height=5)
plot(surv_fit_rho_r, main = "Survival curve", xlab = "pfs_times", ylab = "PFS (%)", col=c(1,2), conf.int = TRUE)
legend("bottomleft", legend = c("rho_r below median", "rho_r above median"), col=c(1,2), lty=1:1)
legend(-1, 0.5, legend = text_info_rho_r, bg="transparent", bty="n")
dev.off()

survdiff(Surv(pfs_times, pfs_events) ~ hi_rho_r, data=data)

# pi r
surv_fit_pi_r = survfit(surv_object ~ strata(hi_pi_r))
# no CI, with text and save as pdf
pdf_file_pi_r = "surv_pi_r.pdf"
pdf(pdf_file_pi_r, width=7,height=5)
plot(surv_fit_pi_r, main = "Survival curve", xlab = "pfs_times", ylab = "PFS (%)", col=c(1,2), conf.int = FALSE)
legend("bottomleft", legend = c("pi_r below median", "pi_r above median"), col=c(1,2), lty=1:1)
cox_pi_r = coxph(Surv(pfs_times, pfs_events) ~ hi_pi_r)
summary_cox_pi_r = summary(cox_pi_r)
pvalue_pi_r = summary_cox_pi_r$coefficients[1, "Pr(>|z|)"]
hazard_ratio_pi_r = summary_cox_pi_r$coefficients[1, "exp(coef)"]
ci_low_pi_r = summary_cox_pi_r$conf.int[1, "lower .95"]
ci_high_pi_r = summary_cox_pi_r$conf.int[1, "upper .95"]
pvalue_text_pi_r = ifelse(pvalue_pi_r < 0.001, "p-value < 0.001", paste("p-value: ", round(pvalue_pi_r, 4)))
text_info_pi_r = paste(pvalue_text_pi_r, 
                        "\nHR: ", round(hazard_ratio_pi_r, 2), 
                        "\n95% CI: (", round(ci_low_pi_r, 2), " - ", round(ci_high_pi_r, 2), ")", sep="")
legend(-1, 0.5, legend = text_info_pi_r, bg="transparent", bty="n")
dev.off()

# with CI plotted
pdf_file_pi_r = "surv_pi_r_CI.pdf"
pdf(pdf_file_pi_r, width=7,height=5)
plot(surv_fit_pi_r, main = "Survival curve", xlab = "pfs_times", ylab = "PFS (%)", col=c(1,2), conf.int = TRUE)
legend("bottomleft", legend = c("pi_r below median", "pi_r above median"), col=c(1,2), lty=1:1)
legend(-1, 0.5, legend = text_info_pi_r, bg="transparent", bty="n")
dev.off()


survdiff(Surv(pfs_times, pfs_events) ~ hi_pi_r, data=data)
