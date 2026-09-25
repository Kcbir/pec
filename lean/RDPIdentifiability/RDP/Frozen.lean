import RDPIdentifiability.RDP.Defs

namespace RDPIdentifiability

open scoped ENNReal

variable {A O Q Q' : Type*}

noncomputable def postWeight (w : ℝ≥0∞) (M : RDP A O Q) (h : List (A × O)) : ℝ≥0∞ :=
  w * M.histLik h

@[simp] lemma postWeight_nil (w : ℝ≥0∞) (M : RDP A O Q) :
    postWeight w M [] = w := by simp [postWeight]

noncomputable def postOdds (w w' : ℝ≥0∞) (M : RDP A O Q) (M' : RDP A O Q')
    (h : List (A × O)) : ℝ≥0∞ :=
  postWeight w M h / postWeight w' M' h

@[simp] lemma postOdds_nil (w w' : ℝ≥0∞) (M : RDP A O Q) (M' : RDP A O Q') :
    postOdds w w' M M' [] = w / w' := by simp [postOdds]

theorem posteriorOdds_eq_priorOdds {π : Policy A O} {M : RDP A O Q} {M' : RDP A O Q'}
    (hEq : ObsEquivOn π M M') (w w' : ℝ≥0∞) {h : List (A × O)}
    (hh : π.histProb h ≠ 0) (hne : M'.histLik h ≠ 0) :
    postOdds w w' M M' h = w / w' := by
  have hfin : M'.histLik h ≠ ⊤ := histLik_ne_top M' h
  rw [postOdds, postWeight, postWeight, hEq h hh]
  rw [ENNReal.mul_div_mul_right _ _ hne hfin]

theorem posteriorOdds_frozen {π : Policy A O} {M : RDP A O Q} {M' : RDP A O Q'}
    (hEq : ObsEquivOn π M M') (w w' : ℝ≥0∞) {h₁ h₂ : List (A × O)}
    (hh₁ : π.histProb h₁ ≠ 0) (hh₂ : π.histProb h₂ ≠ 0)
    (hne₁ : M'.histLik h₁ ≠ 0) (hne₂ : M'.histLik h₂ ≠ 0) :
    postOdds w w' M M' h₁ = postOdds w w' M M' h₂ := by
  rw [posteriorOdds_eq_priorOdds hEq w w' hh₁ hne₁,
      posteriorOdds_eq_priorOdds hEq w w' hh₂ hne₂]

noncomputable def posterior (w w' : ℝ≥0∞) (M : RDP A O Q) (M' : RDP A O Q')
    (h : List (A × O)) : ℝ≥0∞ :=
  postWeight w M h / (postWeight w M h + postWeight w' M' h)

@[simp] lemma posterior_nil (w w' : ℝ≥0∞) (M : RDP A O Q) (M' : RDP A O Q') :
    posterior w w' M M' [] = w / (w + w') := by simp [posterior]

theorem posterior_eq_of_obsEquivOn {π : Policy A O} {M : RDP A O Q} {M' : RDP A O Q'}
    (hEq : ObsEquivOn π M M') (w w' : ℝ≥0∞) {h : List (A × O)}
    (hh : π.histProb h ≠ 0) (hne : M'.histLik h ≠ 0) :
    posterior w w' M M' h = w / (w + w') := by
  have hfin : M'.histLik h ≠ ⊤ := histLik_ne_top M' h
  rw [posterior, postWeight, postWeight, hEq h hh, ← add_mul]
  rw [ENNReal.mul_div_mul_right _ _ hne hfin]

theorem posterior_frozen {π : Policy A O} {M : RDP A O Q} {M' : RDP A O Q'}
    (hEq : ObsEquivOn π M M') (w w' : ℝ≥0∞) {h₁ h₂ : List (A × O)}
    (hh₁ : π.histProb h₁ ≠ 0) (hh₂ : π.histProb h₂ ≠ 0)
    (hne₁ : M'.histLik h₁ ≠ 0) (hne₂ : M'.histLik h₂ ≠ 0) :
    posterior w w' M M' h₁ = posterior w w' M M' h₂ := by
  rw [posterior_eq_of_obsEquivOn hEq w w' hh₁ hne₁,
      posterior_eq_of_obsEquivOn hEq w w' hh₂ hne₂]

theorem frozen_of_obsEquiv {M : RDP A O Q} {M' : RDP A O Q'}
    (hEq : ObsEquiv M M') (w w' : ℝ≥0∞) {h : List (A × O)}
    (hne : M'.histLik h ≠ 0) :
    postOdds w w' M M' h = w / w' := by
  have hfin : M'.histLik h ≠ ⊤ := histLik_ne_top M' h
  rw [postOdds, postWeight, postWeight, hEq h]
  rw [ENNReal.mul_div_mul_right _ _ hne hfin]

end RDPIdentifiability
