import type { Metadata } from "next";
import Link from "next/link";

import { ThresholdExplainer } from "@/components/method/ThresholdExplainer";
import { Code } from "@/components/ui/controls";

export const metadata: Metadata = { title: "Method" };

const SECTIONS = [
  { id: "thesis", label: "Thesis" },
  { id: "pipeline", label: "Pipeline" },
  { id: "retrieval", label: "Retrieval" },
  { id: "generation", label: "Generation" },
  { id: "verification", label: "Verification" },
  { id: "calibration", label: "Calibration" },
  { id: "threshold", label: "Certified threshold" },
  { id: "triage", label: "Safety triage" },
  { id: "limits", label: "Limits" },
  { id: "data", label: "Data" },
] as const;

const STAGES = [
  { name: "Safety triage", does: "A fixed rule list looks for red-flag presentations.", gives: "routine, urgent, or emergency" },
  { name: "Concepts", does: "Terms are matched to a dictionary built from MedlinePlus and MeSH.", gives: "linked concepts and synonyms" },
  { name: "Retrieval", does: "Lexical and dense search run separately, are fused, then reranked.", gives: "ranked passages with their scores" },
  { name: "Generation", does: "The model writes an answer bound to those passages.", gives: "text with numbered citations" },
  { name: "Verification", does: "Each claim is tested against the sentences it cites.", gives: "supported, unsupported, or contradicted" },
  { name: "Confidence", does: "The signals are combined, calibrated, and compared with the threshold.", gives: "release or withhold" },
] as const;

const ANALOGIES = {
  retrieval: {
    text: "Two judges score the same contest, one out of ten and one out of a hundred. You cannot add their marks. You can ask each for a ranking and reward the entries both placed near the top.",
    limit: "the judges here are not independent. Both read the same passage text, so when they agree it is weaker evidence than two independent opinions would be.",
  },
  generation: {
    text: "An open-book exam where the only book allowed is the handful of pages on the desk, and every sentence of the answer must carry the page number it came from.",
    limit: "a model can ignore the instruction. It may state something from memory and attach a citation anyway, which is exactly why the next stage does not take the citations on trust.",
  },
  verification: {
    text: "A fact-checker who does not ask whether a sentence sounds right. They open the cited source, find the relevant line, and check that it says what the writer claims it says.",
    limit: "the checker is itself a model with its own error rate, and it checks whether a claim follows from the passage, not whether the passage is right. A claim faithfully supported by a flawed study is still marked supported.",
  },
  calibration: {
    text: "A weather forecaster is calibrated if it rains on about seven of every ten days for which they said 70%. Being calibrated is not being right every day. It is knowing how often you are right.",
    limit: "a forecaster is calibrated against the weather of the place they forecast for. This model is calibrated on benchmark research questions, and a question of a different kind may not follow the same curve.",
  },
  threshold: {
    text: "An exam board wants at most one in ten of the students who pass to be unprepared. It tries pass marks on a practice exam. A mark that gets exactly one in ten on the practice paper is not safe, since the real cohort will differ, so the board sets the mark a little higher: high enough that it would be surprised, at a stated level, to be wrong.",
    limit: "an exam board can keep testing students until it is sure. Primum has a fixed set of calibration questions, and the guarantee is about the procedure across such sets. It does not promise anything about one particular answer.",
  },
  triage: {
    text: "The laminated card at a clinic reception desk that says which complaints go straight to a nurse. It is not a diagnosis, and nobody pretends it is.",
    limit: "a nurse looks at the patient. This list reads one sentence of text, so it misses emergencies described in unusual words and flags some questions that are only about emergencies.",
  },
} as const;

/**
 * One analogy per idea, with the point at which it stops holding. On wide screens it sits in the
 * margin beside the section it belongs to, the way a journal sets a side note; on narrow screens
 * it follows the first paragraph. The two placements are never visible together.
 */
function Analogy({ name, placement }: { name: keyof typeof ANALOGIES; placement: "margin" | "inline" }) {
  const { text, limit } = ANALOGIES[name];
  const where =
    placement === "margin"
      ? "hidden px-4 py-4 xl:float-right xl:clear-right xl:-mr-[19rem] xl:mb-4 xl:mt-5 xl:block xl:w-[17rem]"
      : "my-6 px-5 py-4 xl:hidden";
  return (
    <aside className={`rounded-card bg-raised font-sans ${where}`}>
      <p className="text-[0.8125rem] font-semibold text-ink">An analogy</p>
      <p className={`mt-1 font-serif text-ink ${placement === "margin" ? "text-[1.0313rem] leading-[1.5]" : "text-[1.125rem] leading-[1.55]"}`}>
        {text}
      </p>
      <p className="mt-2 text-[0.875rem] leading-relaxed text-muted">
        <span className="font-medium text-ink">Where it stops being true: </span>
        {limit}
      </p>
    </aside>
  );
}

function Formula({ children, label }: { children: React.ReactNode; label: string }) {
  return (
    <p role="math" aria-label={label} className="my-5 relative overflow-x-auto whitespace-nowrap font-serif text-[1.25rem] italic leading-loose text-ink">
      {children}
    </p>
  );
}

function H2({ id, children }: { id: string; children: React.ReactNode }) {
  return (
    <h2 id={id} className="scroll-mt-28 pt-12 font-serif text-[1.625rem] font-semibold leading-tight tracking-[-0.012em] text-ink">
      {children}
    </h2>
  );
}

function Built({ children }: { children: React.ReactNode }) {
  return (
    <p className="mt-4 border-l border-control pl-4 font-sans text-[0.875rem] leading-relaxed text-muted">
      <span className="font-medium text-ink">In the code: </span>
      {children}
    </p>
  );
}

export default function MethodPage() {
  return (
    <div className="grid gap-x-12 lg:grid-cols-[11rem_minmax(0,1fr)]">
      <nav aria-label="Sections of this page" className="hidden lg:block">
        <ol className="sticky top-28 space-y-0.5 text-sm">
          {SECTIONS.map((section) => (
            <li key={section.id}>
              <a href={`#${section.id}`} className="block rounded-control py-1.5 text-muted hover:text-ink">
                {section.label}
              </a>
            </li>
          ))}
        </ol>
      </nav>

      <article className="min-w-0">
        <header className="max-w-[46rem]">
          <h1 className="font-serif text-[1.75rem] font-semibold leading-tight tracking-[-0.015em] text-ink">Method</h1>
          <p className="mt-1 font-sans text-[0.9375rem] leading-relaxed text-muted">
            How Primum decides whether to answer, explained without assuming the statistics. Each idea comes with one
            analogy and a note on where the analogy fails.
          </p>
        </header>

        <div className="prose-method flow-root max-w-[46rem] text-ink">
          <H2 id="thesis">The thesis</H2>
          <p className="mt-4 text-[1.375rem] leading-[1.5]">
            A system that answers medical questions from retrieved evidence can be made selectively reliable: it can
            keep the error rate among the answers it gives below a level you choose, by declining to give the rest.
          </p>
          <p className="mt-5">
            <i>Primum non nocere</i> means “first, do no harm”. For a question-answering system the harm is a fluent
            answer that is wrong. Primum does three things to avoid it. It measures how well each claim in an answer
            is supported by the passages it retrieved. It learns a calibrated estimate of the probability that the
            answer is correct. And it sets the line between answering and abstaining with a statistical test that
            makes no assumption about how the data are distributed.
          </p>
          <p>
            The result is a guarantee of a specific kind: among the answers the system releases, the error rate is at
            most α, and this holds with probability at least 1 − δ over the questions used to set the threshold. It is
            a guarantee for PubMedQA benchmark questions answered by the same pipeline, not for every question typed
            into the Ask view. The rest of this page explains each part, and the{" "}
            <Link href="/experiments">Experiments</Link> page tests whether the guarantee holds on questions the
            system has not seen, and how many questions it leaves unanswered.
          </p>

          <H2 id="pipeline">The pipeline</H2>
          <p className="mt-4">
            Every question passes through six stages, in this order. The <Link href="/">Ask</Link> view shows them
            running, with the time each one took.
          </p>
        </div>

        <ol className="mt-6 grid max-w-[62rem] gap-px overflow-hidden rounded-card border border-line bg-(--border) sm:grid-cols-2 lg:grid-cols-3">
          {STAGES.map((stage, i) => (
            <li key={stage.name} className="bg-surface p-4">
              <p className="flex items-baseline gap-2 text-sm font-semibold text-ink">
                <span className="num text-muted">{i + 1}</span>
                {stage.name}
              </p>
              <p className="mt-1 text-sm leading-relaxed text-ink">{stage.does}</p>
              <p className="mt-1.5 text-[0.8125rem] leading-relaxed text-muted">Produces: {stage.gives}</p>
            </li>
          ))}
        </ol>

        <div className="prose-method flow-root max-w-[46rem] text-ink">
          <H2 id="retrieval">Retrieval: two searches, one ranking</H2>
          <Analogy name="retrieval" placement="margin" />
          <p className="mt-4">
            Before anything is written, the system has to find passages worth reading. It searches twice. A lexical
            search (BM25) scores passages by the words they share with the question, weighting rare words more. A
            dense search turns the question and every passage into vectors and looks for passages whose vectors point
            the same way, which finds text that means the same thing in different words. The dense encoder is
            fine-tuned on biomedical question and abstract pairs, so that a question lands near the abstract that
            answers it and away from the others in the same batch.
          </p>
          <Analogy name="retrieval" placement="inline" />
          <p>
            The two searches produce scores on scales that cannot be compared, so the system ignores the scores and
            keeps only the ranks. Reciprocal rank fusion gives each passage a credit from each search:
          </p>
          <Formula label="score of d equals the sum over retrievers r of one over k plus the rank of d in r">
            score(d) = Σ<sub>r</sub> 1 / (k + rank<sub>r</sub>(d))
          </Formula>
          <p>
            A passage near the top of both lists collects two large credits and rises. A passage only one search
            liked still gets through, lower down. A cross-encoder then rereads the fused shortlist with the question
            and reorders it.
          </p>
          <Built>
            BM25 written from scratch on a sparse term-document matrix; a bi-encoder trained with an in-batch-negative
            InfoNCE loss in a hand-written PyTorch loop; exact inner-product search; fusion with k = 60. The{" "}
            <Link href="/retrieval">Retrieval lab</Link> shows the rankings side by side for any query.
          </Built>

          <H2 id="generation">Generation: an answer bound to its sources</H2>
          <Analogy name="generation" placement="margin" />
          <p className="mt-4">
            The language model receives the question and the top passages, numbered. It is told to answer only from
            them, to mark each statement with the number of the passage it came from, and to say what the passages do
            not cover instead of filling the gap from memory. A second call then breaks the finished answer into
            atomic claims, each with its citations, so that every claim can be checked on its own.
          </p>
          <Analogy name="generation" placement="inline" />

          <H2 id="verification">Verification: reading the cited source</H2>
          <Analogy name="verification" placement="margin" />
          <p className="mt-4">
            For each claim, a separate natural language inference model reads the sentences of the passages the claim
            cites and judges whether they entail the claim, contradict it, or do neither. The claim is marked
            supported when some sentence entails it with probability at least one half, contradicted when some
            sentence contradicts it that strongly, and unsupported otherwise. The sentence that decided the verdict
            is kept, and the Ask view shows it under the claim.
          </p>
          <Analogy name="verification" placement="inline" />
          <Built>
            a pretrained cross-encoder run locally; each claim is compared with every sentence and every pair of
            adjacent sentences of its cited passages, and with all passages when it cites none.
          </Built>

          <H2 id="calibration">Calibration: a probability that means what it says</H2>
          <Analogy name="calibration" placement="margin" />
          <p className="mt-4">
            The system now has several signals about the answer: how strong and how consistent the retrieval was, how
            many claims were supported and how firmly, whether anything was contradicted, and how confident the model
            said it was. A small model combines them into one number, the probability that the answer is correct.
            Whether the combination beats the model&apos;s own stated confidence is the second research question; the{" "}
            <Link href="/experiments#rq2">Experiments</Link> page shows the comparison.
          </p>
          <Analogy name="calibration" placement="inline" />
          <p>
            A raw score like that is usually overconfident. Calibration corrects it against held-out questions whose
            answers are known, so that the number can be read literally: of all the answers given a confidence near
            0.8, about four in five should be right.
          </p>
          <Built>
            logistic and one-hidden-layer confidence models in PyTorch; temperature, Platt and isotonic calibration,
            the last with pool-adjacent-violators written from scratch; expected calibration error and Brier score to
            measure the result.
          </Built>

          <H2 id="threshold">The certified threshold: a pass mark with a safety margin</H2>
          <Analogy name="threshold" placement="margin" />
          <p className="mt-4">
            A calibrated confidence still needs a rule: above what value does the system answer? The obvious choice
            is to look at held-out questions, find the lowest threshold at which the answered ones are wrong no more
            than α of the time, and use that. It fails, because the held-out set is a sample. A threshold that just
            meets the target on the sample will often miss it on new questions; the third research question measures how
            often.
          </p>
          <Analogy name="threshold" placement="inline" />
          <p>
            Primum treats each candidate threshold as a hypothesis to be rejected: “at this threshold, the true error
            rate is above α”. Suppose the system answered n calibration questions at a threshold and got k of them
            wrong. If the true error rate really were above α, seeing so few errors would be unlikely, and how
            unlikely is an exact binomial tail:
          </p>
          <Formula label="p equals the probability that a binomial with n trials and success probability alpha is at most k">
            p = P( Bin(n, α) ≤ k )
          </Formula>
          <p>
            When p is at or below δ, the hypothesis is rejected and the threshold is certified. The candidates are
            tested in a fixed order, strictest first, and testing stops at the first one that fails. Because the
            order is fixed before any label is seen and nothing is tested after a failure, the chance that the
            returned threshold is a bad one stays at most δ, however many candidates were tried.
          </p>
          <p>
            Move the risk level below to see the threshold, the share of questions answered, and the error rate among
            them change together.
          </p>
        </div>

        <div className="mt-6 max-w-[62rem]">
          <ThresholdExplainer />
        </div>

        <div className="prose-method flow-root max-w-[46rem] text-ink">
          <H2 id="triage">Safety triage: rules, on purpose</H2>
          <Analogy name="triage" placement="margin" />
          <p className="mt-4">
            Before any of this runs, the question is checked against a short list of red-flag presentations: chest
            pain that spreads or comes with sweating, signs of stroke, anaphylaxis, suicidal intent, overdose, severe
            bleeding, fever in an infant, and others. A match stops the pipeline. The system shows an emergency
            message and answers nothing clinical.
          </p>
          <Analogy name="triage" placement="inline" />
          <p>
            This stage is a list of rules and is documented as one. It is a safety rail, not a research contribution,
            and it is transparent so that anyone can read what triggers it.
          </p>

          <H2 id="limits">What the guarantee does not say</H2>
          <p className="mt-4">Three limits matter when reading any result here.</p>
          <ul className="mt-3 list-disc space-y-3 pl-6">
            <li>
              It is about a rate, not a case. “At most 10% of released answers are wrong” says nothing about whether
              the answer in front of you is one of them.
            </li>
            <li>
              It assumes new questions resemble the calibration questions, in the technical sense that they are
              exchangeable with them. The threshold is set on PubMedQA research questions. A consumer health question
              asked in the Ask view is a different kind of question, so there the threshold is a principled default,
              not a proven bound.
            </li>
            <li>
              “Correct” means agreeing with the expert label of a benchmark. It is not a claim about clinical
              outcomes, and this system is a research prototype, not a source of medical advice.
            </li>
          </ul>

          <H2 id="data">Data</H2>
          <p className="mt-4">All corpora are openly licensed and downloaded by code.</p>
          <ul className="mt-3 list-disc space-y-3 pl-6">
            <li>
              <a href="https://huggingface.co/datasets/qiaojin/PubMedQA" target="_blank" rel="noreferrer noopener">
                PubMedQA
              </a>
              : the expert-labelled subset of yes, no or maybe research questions, each with its PubMed abstract. Used for
              the confidence and abstention experiments. The abstract’s conclusion is never indexed, because it would
              leak the label.
            </li>
            <li>
              <a href="https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/" target="_blank" rel="noreferrer noopener">
                NFCorpus
              </a>
              , in its BEIR form: a medical retrieval benchmark with graded relevance judgements. Used to compare
              retrievers.
            </li>
            <li>
              <a href="https://medlineplus.gov/xml.html" target="_blank" rel="noreferrer noopener">
                MedlinePlus health topics
              </a>
              , from the US National Library of Medicine: the knowledge base behind live answers, and the source of
              the concept dictionary.
            </li>
          </ul>
          <p>
            Every experiment is re-run with <Code>primum exp all</Code>, and every figure in this interface is drawn
            from the files that command writes.
          </p>
        </div>
      </article>
    </div>
  );
}
