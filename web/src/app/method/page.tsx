import type { Metadata } from "next";
import Link from "next/link";

import { SECTIONS } from "@/components/method/content";
import { Doc, LimitItem, LimitList, PipelineStages, SourceList, Term, TermList } from "@/components/method/doc";
import { OnThisPage } from "@/components/method/OnThisPage";
import { ThresholdExplainer } from "@/components/method/ThresholdExplainer";
import { buttonClass } from "@/components/ui/button";
import { Code } from "@/components/ui/code";
import { PageHeader } from "@/components/ui/page-header";
import { ArrowRightGlyph } from "@/components/ui/glyphs";

export const metadata: Metadata = { title: "Method" };

export default function MethodPage() {
  return (
    <div className="grid gap-x-10 gap-y-6 lg:grid-cols-[12rem_minmax(0,1fr)] xl:gap-x-12">
      <div className="lg:col-start-2">
        <PageHeader>
          <PageHeader.Heading className="basis-[26rem]">
            <PageHeader.Title>Method</PageHeader.Title>
            <PageHeader.Description>
              How Primum decides whether to answer, explained without assuming the statistics. Each idea comes with one
              analogy and a note on where the analogy fails.
            </PageHeader.Description>
          </PageHeader.Heading>
          <PageHeader.Actions>
            <Link href="/experiments" className={buttonClass("secondary", "sm")}>
              See the experiments
              <ArrowRightGlyph size={14} />
            </Link>
          </PageHeader.Actions>
        </PageHeader>
      </div>

      <div className="lg:col-start-1 lg:row-span-2 lg:row-start-1">
        <OnThisPage sections={SECTIONS} />
      </div>

      <Doc className="lg:col-start-2 lg:pt-4">
        <article>
          <Doc.Section id="thesis" title="The thesis">
            <Doc.Lead>
              A system that answers medical questions from retrieved evidence can be made selectively reliable: it can
              keep the error rate among the answers it gives below a level you choose, by declining to give the rest.
            </Doc.Lead>
            <Doc.Note label="Notation on this page">
              <Doc.Note.Body className="pt-2.5">
                <TermList>
                  <Term name="α">The tolerated error rate among released answers. You choose it.</Term>
                  <Term name="δ">The largest chance that the procedure returns a threshold that breaks α.</Term>
                  <Term name="n, k">Calibration questions answered at a threshold, and how many of them were wrong.</Term>
                </TermList>
              </Doc.Note.Body>
              <Doc.Note.Footer>
                <p>
                  <span className="font-medium text-ink">Withheld </span>
                  answers are abstentions, not errors. They count against coverage, never against α.
                </p>
              </Doc.Note.Footer>
            </Doc.Note>
            <p>
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
              <Link href="/experiments">Experiments</Link> page tests whether the guarantee holds on questions the system
              has not seen, and how many questions it leaves unanswered.
            </p>
          </Doc.Section>

          <Doc.Section id="pipeline" title="The pipeline">
            <p>
              Every question passes through six stages, in this order. The <Link href="/">Ask</Link> view shows them
              running, with the time each one took.
            </p>
            <Doc.Wide className="mt-6">
              <PipelineStages />
            </Doc.Wide>
          </Doc.Section>

          <Doc.Section id="retrieval" title="Retrieval: two searches, one ranking">
            <p>
              Before anything is written, the system has to find passages worth reading. It searches twice. A lexical
              search (BM25) scores passages by the words they share with the question, weighting rare words more. A
              dense search turns the question and every passage into vectors and looks for passages whose vectors point
              the same way, which finds text that means the same thing in different words. The dense encoder is
              fine-tuned on biomedical question and abstract pairs, so that a question lands near the abstract that
              answers it and away from the others in the same batch.
            </p>
            <Doc.Analogy name="retrieval" />
            <p>
              The two searches produce scores on scales that cannot be compared, so the system ignores the scores and
              keeps only the ranks. Reciprocal rank fusion gives each passage a credit from each search:
            </p>
            <Doc.Formula
              caption="Reciprocal rank fusion"
              label="score of d equals the sum over retrievers r of one over k plus the rank of d in r"
            >
              score(d) = Σ<sub>r</sub> 1 / (k + rank<sub>r</sub>(d))
            </Doc.Formula>
            <p>
              A passage near the top of both lists collects two large credits and rises. A passage only one search liked
              still gets through, lower down. A cross-encoder then rereads the fused shortlist with the question and
              reorders it.
            </p>
            <Doc.Built>
              BM25 written from scratch on a sparse term-document matrix; a bi-encoder trained with an in-batch-negative
              InfoNCE loss in a hand-written PyTorch loop; exact inner-product search; fusion with k = 60. The{" "}
              <Link href="/retrieval">Retrieval lab</Link> shows the rankings side by side for any query.
            </Doc.Built>
          </Doc.Section>

          <Doc.Section id="generation" title="Generation: an answer bound to its sources">
            <p>
              The language model receives the question and the top passages, numbered. It is told to answer only from
              them, to mark each statement with the number of the passage it came from, and to say what the passages do
              not cover instead of filling the gap from memory. A second call then breaks the finished answer into atomic
              claims, each with its citations, so that every claim can be checked on its own.
            </p>
            <Doc.Analogy name="generation" />
          </Doc.Section>

          <Doc.Section id="verification" title="Verification: reading the cited source">
            <p>
              For each claim, a separate natural language inference model reads the sentences of the passages the claim
              cites and judges whether they entail the claim, contradict it, or do neither. The claim is marked supported
              when some sentence entails it with probability at least one half, contradicted when some sentence
              contradicts it that strongly, and unsupported otherwise. The sentence that decided the verdict is kept, and
              the Ask view shows it under the claim.
            </p>
            <Doc.Analogy name="verification" />
            <Doc.Built>
              a pretrained cross-encoder run locally; each claim is compared with every sentence and every pair of
              adjacent sentences of its cited passages, and with all passages when it cites none.
            </Doc.Built>
          </Doc.Section>

          <Doc.Section id="calibration" title="Calibration: a probability that means what it says">
            <p>
              The system now has several signals about the answer: how strong and how consistent the retrieval was, how
              many claims were supported and how firmly, whether anything was contradicted, and how confident the model
              said it was. A small model combines them into one number, the probability that the answer is correct.
              Whether the combination beats the model&apos;s own stated confidence is the second research question; the{" "}
              <Link href="/experiments#rq2">Experiments</Link> page shows the comparison.
            </p>
            <Doc.Analogy name="calibration" />
            <p>
              A raw score like that is usually overconfident. Calibration corrects it against held-out questions whose
              answers are known, so that the number can be read literally: of all the answers given a confidence near
              0.8, about four in five should be right.
            </p>
            <Doc.Built>
              logistic and one-hidden-layer confidence models in PyTorch; temperature, Platt and isotonic calibration,
              the last with pool-adjacent-violators written from scratch; expected calibration error and Brier score to
              measure the result.
            </Doc.Built>
          </Doc.Section>

          <Doc.Section id="threshold" title="The certified threshold: a pass mark with a safety margin">
            <p>
              A calibrated confidence still needs a rule: above what value does the system answer? The obvious choice is
              to look at held-out questions, find the lowest threshold at which the answered ones are wrong no more than
              α of the time, and use that. It fails, because the held-out set is a sample. A threshold that just meets
              the target on the sample will often miss it on new questions; the third research question measures how
              often.
            </p>
            <Doc.Analogy name="threshold" />
            <p>
              Primum treats each candidate threshold as a hypothesis to be rejected: “at this threshold, the true error
              rate is above α”. Suppose the system answered n calibration questions at a threshold and got k of them
              wrong. If the true error rate really were above α, seeing so few errors would be unlikely, and how unlikely
              is an exact binomial tail:
            </p>
            <Doc.Formula
              caption="Exact binomial tail"
              label="p equals the probability that a binomial with n trials and success probability alpha is at most k"
            >
              p = P( Bin(n, α) ≤ k )
            </Doc.Formula>
            <p>
              When p is at or below δ, the hypothesis is rejected and the threshold is certified. The candidates are
              tested in a fixed order, strictest first, and testing stops at the first one that fails. Because the order
              is fixed before any label is seen and nothing is tested after a failure, the chance that the returned
              threshold is a bad one stays at most δ, however many candidates were tried.
            </p>
            <p>
              Move the risk level below to see the threshold, the share of questions answered, and the error rate among
              them change together.
            </p>
            <Doc.Wide>
              <ThresholdExplainer />
            </Doc.Wide>
          </Doc.Section>

          <Doc.Section id="triage" title="Safety triage: rules, on purpose">
            <p>
              Before any of this runs, the question is checked against a short list of red-flag presentations: chest
              pain that spreads or comes with sweating, signs of stroke, anaphylaxis, suicidal intent, overdose, severe
              bleeding, fever in an infant, and others. A match stops the pipeline. The system shows an emergency message
              and answers nothing clinical.
            </p>
            <Doc.Analogy name="triage" />
            <p>
              This stage is a list of rules and is documented as one. It is a safety rail, not a research contribution,
              and it is transparent so that anyone can read what triggers it.
            </p>
          </Doc.Section>

          <Doc.Section id="limits" title="What the guarantee does not say">
            <p>Three limits matter when reading any result here.</p>
            <LimitList>
              <LimitItem>
                It is about a rate, not a case. “At most 10% of released answers are wrong” says nothing about whether
                the answer in front of you is one of them.
              </LimitItem>
              <LimitItem>
                It assumes new questions resemble the calibration questions, in the technical sense that they are
                exchangeable with them. The threshold is set on PubMedQA research questions. A consumer health question
                asked in the Ask view is a different kind of question, so there the threshold is a principled default,
                not a proven bound.
              </LimitItem>
              <LimitItem>
                “Correct” means agreeing with the expert label of a benchmark. It is not a claim about clinical outcomes,
                and this system is a research prototype, not a source of medical advice.
              </LimitItem>
            </LimitList>
          </Doc.Section>

          <Doc.Section id="data" title="Data">
            <p>All corpora are openly licensed and downloaded by code.</p>
            <SourceList />
            <p>
              Every experiment is re-run with <Code>primum exp all</Code>, and every figure in this interface is drawn
              from the files that command writes.
            </p>
          </Doc.Section>
        </article>
      </Doc>
    </div>
  );
}
