import Mathlib

namespace RDPIdentifiability

open scoped ENNReal

structure RDP (A O Q : Type*) where
  start : Q
  step : Q → A → O → Q
  emit : Q → A → O → ℝ≥0∞
  emit_sum : ∀ q a, ∑' o, emit q a o = 1

variable {A O Q Q' : Type*}

def RDP.traverseFrom (M : RDP A O Q) : Q → List (A × O) → Q
  | q, [] => q
  | q, (a, o) :: h => M.traverseFrom (M.step q a o) h

def RDP.traverse (M : RDP A O Q) (h : List (A × O)) : Q :=
  M.traverseFrom M.start h

@[simp] lemma RDP.traverseFrom_nil (M : RDP A O Q) (q : Q) :
    M.traverseFrom q [] = q := rfl

@[simp] lemma RDP.traverseFrom_cons (M : RDP A O Q) (q : Q) (a : A) (o : O)
    (h : List (A × O)) :
    M.traverseFrom q ((a, o) :: h) = M.traverseFrom (M.step q a o) h := rfl

@[simp] lemma RDP.traverse_nil (M : RDP A O Q) : M.traverse [] = M.start := rfl

noncomputable def RDP.histLikFrom (M : RDP A O Q) : Q → List (A × O) → ℝ≥0∞
  | _, [] => 1
  | q, (a, o) :: h => M.emit q a o * M.histLikFrom (M.step q a o) h

noncomputable def RDP.histLik (M : RDP A O Q) (h : List (A × O)) : ℝ≥0∞ :=
  M.histLikFrom M.start h

@[simp] lemma RDP.histLikFrom_nil (M : RDP A O Q) (q : Q) :
    M.histLikFrom q [] = 1 := rfl

@[simp] lemma RDP.histLikFrom_cons (M : RDP A O Q) (q : Q) (a : A) (o : O)
    (h : List (A × O)) :
    M.histLikFrom q ((a, o) :: h) = M.emit q a o * M.histLikFrom (M.step q a o) h := rfl

@[simp] lemma RDP.histLik_nil (M : RDP A O Q) : M.histLik [] = 1 := rfl

lemma RDP.histLik_cons (M : RDP A O Q) (a : A) (o : O) (h : List (A × O)) :
    M.histLik ((a, o) :: h)
      = M.emit M.start a o * M.histLikFrom (M.step M.start a o) h := rfl

lemma RDP.emit_le_one (M : RDP A O Q) (q : Q) (a : A) (o : O) : M.emit q a o ≤ 1 := by
  calc M.emit q a o ≤ ∑' o', M.emit q a o' := ENNReal.le_tsum o
    _ = 1 := M.emit_sum q a

lemma RDP.histLikFrom_le_one (M : RDP A O Q) (q : Q) (h : List (A × O)) :
    M.histLikFrom q h ≤ 1 := by
  induction h generalizing q with
  | nil => simp
  | cons ao t ih =>
      obtain ⟨a, o⟩ := ao
      rw [RDP.histLikFrom_cons]
      exact mul_le_one' (M.emit_le_one q a o) (ih _)

lemma RDP.histLik_le_one (M : RDP A O Q) (h : List (A × O)) : M.histLik h ≤ 1 :=
  M.histLikFrom_le_one M.start h

lemma histLik_ne_top (M : RDP A O Q) (h : List (A × O)) : M.histLik h ≠ ⊤ :=
  ne_top_of_le_ne_top ENNReal.one_ne_top (M.histLik_le_one h)

structure Policy (A O : Type*) where
  prob : List (A × O) → A → ℝ≥0∞
  prob_sum : ∀ h, ∑' a, prob h a = 1

noncomputable def Policy.histProbFrom (π : Policy A O) :
    List (A × O) → List (A × O) → ℝ≥0∞
  | _, [] => 1
  | acc, (a, o) :: h => π.prob acc a * π.histProbFrom (acc ++ [(a, o)]) h

noncomputable def Policy.histProb (π : Policy A O) (h : List (A × O)) : ℝ≥0∞ :=
  π.histProbFrom [] h

@[simp] lemma Policy.histProbFrom_nil (π : Policy A O) (acc : List (A × O)) :
    π.histProbFrom acc [] = 1 := rfl

@[simp] lemma Policy.histProbFrom_cons (π : Policy A O) (acc : List (A × O))
    (a : A) (o : O) (h : List (A × O)) :
    π.histProbFrom acc ((a, o) :: h)
      = π.prob acc a * π.histProbFrom (acc ++ [(a, o)]) h := rfl

@[simp] lemma Policy.histProb_nil (π : Policy A O) : π.histProb [] = 1 := rfl

lemma Policy.histProb_cons (π : Policy A O) (a : A) (o : O) (h : List (A × O)) :
    π.histProb ((a, o) :: h) = π.prob [] a * π.histProbFrom [(a, o)] h := rfl

noncomputable def pathProb (π : Policy A O) (M : RDP A O Q) (h : List (A × O)) : ℝ≥0∞ :=
  π.histProb h * M.histLik h

lemma pathProb_eq (π : Policy A O) (M : RDP A O Q) (h : List (A × O)) :
    pathProb π M h = π.histProb h * M.histLik h := rfl

def ObsEquiv (M : RDP A O Q) (M' : RDP A O Q') : Prop :=
  ∀ h : List (A × O), M.histLik h = M'.histLik h

namespace ObsEquiv

variable {M : RDP A O Q} {M' : RDP A O Q'}

lemma refl (M : RDP A O Q) : ObsEquiv M M := fun _ => rfl

lemma symm (hEq : ObsEquiv M M') : ObsEquiv M' M := fun h => (hEq h).symm

lemma trans {Q'' : Type*} {M'' : RDP A O Q''}
    (h₁ : ObsEquiv M M') (h₂ : ObsEquiv M' M'') : ObsEquiv M M'' :=
  fun h => (h₁ h).trans (h₂ h)

lemma pathProb_eq (hEq : ObsEquiv M M') (π : Policy A O) (h : List (A × O)) :
    pathProb π M h = pathProb π M' h := by
  simp [RDPIdentifiability.pathProb, hEq h]

end ObsEquiv

def ObsEquivOn (π : Policy A O) (M : RDP A O Q) (M' : RDP A O Q') : Prop :=
  ∀ h : List (A × O), π.histProb h ≠ 0 → M.histLik h = M'.histLik h

lemma ObsEquiv.obsEquivOn {M : RDP A O Q} {M' : RDP A O Q'}
    (hEq : ObsEquiv M M') (π : Policy A O) : ObsEquivOn π M M' :=
  fun h _ => hEq h

namespace ObsEquivOn

variable {π : Policy A O} {M : RDP A O Q} {M' : RDP A O Q'}

lemma refl (π : Policy A O) (M : RDP A O Q) : ObsEquivOn π M M := fun _ _ => rfl

lemma symm (hEq : ObsEquivOn π M M') : ObsEquivOn π M' M :=
  fun h hh => (hEq h hh).symm

lemma trans {Q'' : Type*} {M'' : RDP A O Q''}
    (h₁ : ObsEquivOn π M M') (h₂ : ObsEquivOn π M' M'') : ObsEquivOn π M M'' :=
  fun h hh => (h₁ h hh).trans (h₂ h hh)

lemma pathProb_eq (hEq : ObsEquivOn π M M') (h : List (A × O)) :
    pathProb π M h = pathProb π M' h := by
  by_cases hh : π.histProb h = 0
  · simp [RDPIdentifiability.pathProb, hh]
  · simp [RDPIdentifiability.pathProb, hEq h hh]

end ObsEquivOn

end RDPIdentifiability
