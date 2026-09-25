import RDPIdentifiability.RDP.Frozen

namespace RDPIdentifiability

open scoped ENNReal

variable {A O Q Q' : Type*}

lemma RDP.histLikFrom_append (M : RDP A O Q) (q : Q) (h : List (A × O)) (a : A) (o : O) :
    M.histLikFrom q (h ++ [(a, o)])
      = M.histLikFrom q h * M.emit (M.traverseFrom q h) a o := by
  induction h generalizing q with
  | nil => simp [RDP.histLikFrom]
  | cons ao t ih =>
      obtain ⟨a', o'⟩ := ao
      rw [List.cons_append, RDP.histLikFrom_cons, RDP.histLikFrom_cons,
          RDP.traverseFrom_cons, ih, mul_assoc]

lemma RDP.histLik_append (M : RDP A O Q) (h : List (A × O)) (a : A) (o : O) :
    M.histLik (h ++ [(a, o)]) = M.histLik h * M.emit (M.traverse h) a o :=
  M.histLikFrom_append M.start h a o

lemma RDP.tsum_histLik_append (M : RDP A O Q) (h : List (A × O)) (a : A) :
    ∑' o, M.histLik (h ++ [(a, o)]) = M.histLik h := by
  simp_rw [M.histLik_append h a]
  rw [ENNReal.tsum_mul_left, M.emit_sum, mul_one]

theorem tsum_postWeight_append (w : ℝ≥0∞) (M : RDP A O Q) (h : List (A × O)) (a : A) :
    ∑' o, postWeight w M (h ++ [(a, o)]) = postWeight w M h := by
  simp_rw [postWeight]
  rw [ENNReal.tsum_mul_left, M.tsum_histLik_append h a]

noncomputable def evidence (w w' : ℝ≥0∞) (M : RDP A O Q) (M' : RDP A O Q')
    (h : List (A × O)) : ℝ≥0∞ :=
  postWeight w M h + postWeight w' M' h

theorem tsum_evidence_append (w w' : ℝ≥0∞) (M : RDP A O Q) (M' : RDP A O Q')
    (h : List (A × O)) (a : A) :
    ∑' o, evidence w w' M M' (h ++ [(a, o)]) = evidence w w' M M' h := by
  simp_rw [evidence]
  rw [ENNReal.tsum_add, tsum_postWeight_append, tsum_postWeight_append]

theorem posterior_le_one (w w' : ℝ≥0∞) (M : RDP A O Q) (M' : RDP A O Q')
    (h : List (A × O)) : posterior w w' M M' h ≤ 1 := by
  rw [posterior]
  exact ENNReal.div_le_of_le_mul (by simp)

theorem evidence_mul_posterior (w w' : ℝ≥0∞) (M : RDP A O Q) (M' : RDP A O Q')
    (h : List (A × O)) (hne : evidence w w' M M' h ≠ 0)
    (htop : evidence w w' M M' h ≠ ⊤) :
    evidence w w' M M' h * posterior w w' M M' h = postWeight w M h := by
  have hev : evidence w w' M M' h = postWeight w M h + postWeight w' M' h := rfl
  rw [posterior, mul_comm, ← hev]
  exact ENNReal.div_mul_cancel hne htop

theorem posterior_tower (w w' : ℝ≥0∞) (M : RDP A O Q) (M' : RDP A O Q')
    (h : List (A × O)) (a : A)
    (hstep : ∀ o : O, evidence w w' M M' (h ++ [(a, o)]) ≠ 0)
    (hstep' : ∀ o : O, evidence w w' M M' (h ++ [(a, o)]) ≠ ⊤) :
    ∑' o, evidence w w' M M' (h ++ [(a, o)]) * posterior w w' M M' (h ++ [(a, o)])
      = postWeight w M h := by
  have : ∀ o : O, evidence w w' M M' (h ++ [(a, o)]) * posterior w w' M M' (h ++ [(a, o)])
      = postWeight w M (h ++ [(a, o)]) :=
    fun o => evidence_mul_posterior w w' M M' _ (hstep o) (hstep' o)
  simp_rw [this]
  exact tsum_postWeight_append w M h a

end RDPIdentifiability
