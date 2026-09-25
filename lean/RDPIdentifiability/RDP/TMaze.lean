import RDPIdentifiability.RDP.Frozen

namespace RDPIdentifiability

open scoped ENNReal

inductive Act
  | wait
  | run
  deriving DecidableEq

inductive Obs
  | left
  | right
  deriving DecidableEq

inductive CueState
  | init
  | sawLeft
  | sawRight
  deriving DecidableEq

instance : Fintype Act := ⟨{Act.wait, Act.run}, by intro x; cases x <;> simp⟩
instance : Fintype Obs := ⟨{Obs.left, Obs.right}, by intro x; cases x <;> simp⟩
instance : Fintype CueState :=
  ⟨{CueState.init, CueState.sawLeft, CueState.sawRight}, by intro x; cases x <;> simp⟩

open Act Obs CueState

lemma tsum_obs (f : Obs → ℝ≥0∞) : ∑' o, f o = f left + f right := by
  rw [tsum_fintype]
  show ∑ o ∈ ({left, right} : Finset Obs), f o = _
  rw [Finset.sum_insert (by simp), Finset.sum_singleton]

noncomputable def cueEmit : CueState → Act → Obs → ℝ≥0∞
  | init, _, _ => 1 / 2
  | sawLeft, wait, _ => 1 / 2
  | sawRight, wait, _ => 1 / 2
  | sawLeft, run, left => 1
  | sawLeft, run, right => 0
  | sawRight, run, left => 0
  | sawRight, run, right => 1

lemma cueEmit_sum (q : CueState) (a : Act) : ∑' o, cueEmit q a o = 1 := by
  rw [tsum_obs]
  cases q <;> cases a <;> simp [cueEmit, ENNReal.inv_two_add_inv_two]

noncomputable def cueModel : RDP Act Obs CueState where
  start := init
  step := fun q _ o => match q with
    | init => match o with
      | left => sawLeft
      | right => sawRight
    | sawLeft => sawLeft
    | sawRight => sawRight
  emit := cueEmit
  emit_sum := cueEmit_sum

noncomputable def blindModel : RDP Act Obs Unit where
  start := ()
  step := fun _ _ _ => ()
  emit := fun _ _ _ => 1 / 2
  emit_sum := by
    intro q a
    rw [tsum_obs]
    exact ENNReal.add_halves 1

@[simp] lemma blindModel_histLikFrom (u : Unit) (h : List (Act × Obs)) :
    blindModel.histLikFrom u h = (1 / 2) ^ h.length := by
  induction h generalizing u with
  | nil => simp
  | cons ao t ih =>
      obtain ⟨a, o⟩ := ao
      rw [RDP.histLikFrom_cons, ih]
      show (1 / 2 : ℝ≥0∞) * (1 / 2) ^ t.length = (1 / 2) ^ (t.length + 1)
      rw [pow_succ, mul_comm]

lemma blindModel_histLik (h : List (Act × Obs)) :
    blindModel.histLik h = (1 / 2) ^ h.length := by
  simp [RDP.histLik]

def AllWait : List (Act × Obs) → Prop :=
  fun h => ∀ p ∈ h, p.1 = wait

@[simp] lemma allWait_nil : AllWait ([] : List (Act × Obs)) := by
  intro p hp; simp at hp

lemma AllWait.tail {a : Act} {o : Obs} {t : List (Act × Obs)}
    (h : AllWait ((a, o) :: t)) : AllWait t :=
  fun p hp => h p (List.mem_cons_of_mem _ hp)

lemma AllWait.head {a : Act} {o : Obs} {t : List (Act × Obs)}
    (h : AllWait ((a, o) :: t)) : a = wait :=
  h (a, o) (List.mem_cons_self ..)

lemma cueModel_histLikFrom_allWait (q : CueState) {h : List (Act × Obs)}
    (hw : AllWait h) : cueModel.histLikFrom q h = (1 / 2) ^ h.length := by
  induction h generalizing q with
  | nil => simp
  | cons ao t ih =>
      obtain ⟨a, o⟩ := ao
      have ha : a = wait := hw.head
      subst ha
      rw [RDP.histLikFrom_cons, ih _ hw.tail]
      have hemit : cueModel.emit q wait o = (1 / 2 : ℝ≥0∞) := by
        cases q <;> cases o <;> rfl
      rw [hemit]
      show (1 / 2 : ℝ≥0∞) * (1 / 2) ^ t.length = (1 / 2) ^ (t.length + 1)
      rw [pow_succ, mul_comm]

lemma cueModel_histLik_allWait {h : List (Act × Obs)} (hw : AllWait h) :
    cueModel.histLik h = (1 / 2) ^ h.length :=
  cueModel_histLikFrom_allWait _ hw

theorem obsEquivOn_waitPolicy {h : List (Act × Obs)} (hw : AllWait h) :
    cueModel.histLik h = blindModel.histLik h := by
  rw [cueModel_histLik_allWait hw, blindModel_histLik]

def separatingHistory : List (Act × Obs) := [(wait, left), (run, right)]

lemma cueModel_separating : cueModel.histLik separatingHistory = 0 := by
  show cueEmit init wait left * (cueEmit sawLeft run right * 1) = 0
  norm_num [cueEmit]

lemma blindModel_separating : blindModel.histLik separatingHistory = 1 / 4 := by
  rw [blindModel_histLik]
  show ((1 : ℝ≥0∞) / 2) ^ 2 = 1 / 4
  rw [one_div, ← ENNReal.inv_pow]
  norm_num

theorem not_obsEquiv : ¬ ObsEquiv cueModel blindModel := by
  intro hEq
  have h := hEq separatingHistory
  rw [cueModel_separating, blindModel_separating] at h
  exact (ENNReal.div_ne_zero.mpr ⟨one_ne_zero, by norm_num⟩) h.symm

theorem separating_history_offPolicy :
    ¬ AllWait separatingHistory
      ∧ cueModel.histLik separatingHistory ≠ blindModel.histLik separatingHistory := by
  refine ⟨fun hw => ?_, ?_⟩
  · have := hw (run, right) (by simp [separatingHistory])
    exact Act.noConfusion this
  · rw [cueModel_separating, blindModel_separating]
    exact fun hcontra =>
      (ENNReal.div_ne_zero.mpr ⟨one_ne_zero, by norm_num⟩) hcontra.symm

theorem collapse_test :
    (∀ h : List (Act × Obs), AllWait h →
        cueModel.histLik h = blindModel.histLik h)
      ∧ ¬ ObsEquiv cueModel blindModel :=
  ⟨fun _ hw => obsEquivOn_waitPolicy hw, not_obsEquiv⟩

theorem waitPolicy_covers_states :
    cueModel.traverse [(wait, left)] = sawLeft
      ∧ cueModel.traverse [(wait, right)] = sawRight
      ∧ cueModel.histLik [(wait, left)] ≠ 0
      ∧ cueModel.histLik [(wait, right)] ≠ 0 := by
  refine ⟨rfl, rfl, ?_, ?_⟩
  · show cueEmit init wait left * 1 ≠ 0
    norm_num [cueEmit]
  · show cueEmit init wait right * 1 ≠ 0
    norm_num [cueEmit]

noncomputable def waitPolicy : Policy Act Obs where
  prob := fun _ a => if a = wait then 1 else 0
  prob_sum := by
    intro h
    rw [tsum_fintype]
    show ∑ a ∈ ({wait, run} : Finset Act), (if a = wait then (1 : ℝ≥0∞) else 0) = 1
    rw [Finset.sum_insert (by simp), Finset.sum_singleton]
    norm_num

lemma waitPolicy_histProbFrom_ne_zero (acc : List (Act × Obs)) {h : List (Act × Obs)}
    (hw : AllWait h) : waitPolicy.histProbFrom acc h ≠ 0 := by
  induction h generalizing acc with
  | nil => simp
  | cons ao t ih =>
      obtain ⟨a, o⟩ := ao
      have ha : a = wait := hw.head
      subst ha
      rw [Policy.histProbFrom_cons]
      refine mul_ne_zero ?_ (ih _ hw.tail)
      show (if wait = wait then (1 : ℝ≥0∞) else 0) ≠ 0
      norm_num

lemma allWait_of_histProbFrom_ne_zero (acc : List (Act × Obs)) {h : List (Act × Obs)}
    (hp : waitPolicy.histProbFrom acc h ≠ 0) : AllWait h := by
  induction h generalizing acc with
  | nil => simp
  | cons ao t ih =>
      obtain ⟨a, o⟩ := ao
      rw [Policy.histProbFrom_cons] at hp
      have hfac := mul_ne_zero_iff.mp hp
      have ha : a = wait := by
        by_contra hne
        apply hfac.1
        show (if a = wait then (1 : ℝ≥0∞) else 0) = 0
        exact ite_eq_right_iff.mpr (fun hc => absurd hc hne)
      subst ha
      intro p hpm
      rcases List.mem_cons.mp hpm with rfl | hmem
      · rfl
      · exact ih _ hfac.2 p hmem

theorem tmaze_obsEquivOn : ObsEquivOn waitPolicy cueModel blindModel := by
  intro h hh
  exact obsEquivOn_waitPolicy (allWait_of_histProbFrom_ne_zero [] hh)

theorem tmaze_posterior_frozen (w w' : ℝ≥0∞) {h : List (Act × Obs)}
    (hw : AllWait h) :
    postOdds w w' cueModel blindModel h = w / w' := by
  have hne : blindModel.histLik h ≠ 0 := by
    rw [blindModel_histLik]
    exact pow_ne_zero _ (by norm_num)
  have hfin : blindModel.histLik h ≠ ⊤ := histLik_ne_top blindModel h
  rw [postOdds, postWeight, postWeight, obsEquivOn_waitPolicy hw]
  rw [ENNReal.mul_div_mul_right _ _ hne hfin]

theorem tmaze_frozen_forever (w w' : ℝ≥0∞) {h₁ h₂ : List (Act × Obs)}
    (hw₁ : AllWait h₁) (hw₂ : AllWait h₂) :
    postOdds w w' cueModel blindModel h₁ = postOdds w w' cueModel blindModel h₂ := by
  rw [tmaze_posterior_frozen w w' hw₁, tmaze_posterior_frozen w w' hw₂]

end RDPIdentifiability
