You asked:
----------

My goal is state of the art object detection and classification of small targets in very large satellite images (e.g. 50000x100000 pixels). Data available to me: I have thousands of PDF documents containing domain information about objects from these images e.g. annotated image chips and accompanying graphics and text. I've got around 30TB of example images, however, the objects of interest tend to be relatively rare e.g. there might be lots of cars in a satellite image, but they only make up a small percentage of the total pixels. Some object might be rare, even across images, e.g. only occurring in 1 in 100, or 1 in 1000 images. Additionally, all images have geocoding, time and sensor positioning data. It would be good to take advantage of this (assuming it can provide benefit), as I don't think I've ever seen it utilised in automated object detection and classification. I have a small number of labeled images, maybe 100 with about 30,000 labels. Compute resources: I have 128 80GB A100 GPU's (4 GPU's per node) that I can use. No particular time limit on how long I can run compute for, but would like at least initial results after at most 2 weeks. Some initial thoughts: I'd like to make use of the vast amount of unlabeled imagery and PDF domain specific information. I'd like to move away from the purely supervised learning approach I've used to date. e.g. Taking an imagenet pre-trained model like resnet50 with a MaskRCNN head and trying to fine tune on labelled domain data. This does yield good results, but I'm forever trying to broaden the distribution by adding training data to improve performance. This isn't scalable. Utilising foundational transformer models seems like a much better approach. The current approach also chooses a small number of target classes of interest, then attempts to detect and classify them whenever they occur in images. This locks you into choice of detector upfront. Adding classes is difficult, and requires reprocessing if you want to find things in old images. I wonder if there are approaches where you could efficiently store the image content (e.g. embeddings) given most of the image doesn't contain objects of interest, and define your objects of interest after the fact? If this is infeasible, or sub-optimal I'd still be open to a more traditional object detector approach.

---

ChatGPT Replied:
----------------

Thanks for the detailed context. To tailor the best object detection and classification pipeline for your satellite imagery, could you clarify a few things:

1.  What types of objects are you most interested in detecting (e.g. vehicles, buildings, ships, rare infrastructure)?
    
2.  Are your PDF documents machine-readable, or do they contain scanned images of text and annotations?
    
3.  Are there specific geospatial or temporal patterns you'd like to leverage (e.g. object types more common in certain locations or seasons)?
    
4.  Would you prefer a system capable of few-shot or zero-shot object detection based on textual or visual queries?
    
5.  Are you interested in integrating multi-modal learning, combining text from PDFs with satellite images?
    

Once I have this information, I’ll get started on an in-depth research plan.

---

You asked:
----------

1\. Traditionally, I've been primarily interested in vehicles (ships and planes), but that's more down to the limitations of previous approaches (it was a lot of effort to label and train for specific targets). I'd certainly be interested in the ability to easily find new targets of interest e.g. "specific buildings that look like this", or describing rare objects (perhaps with text, or photographs). 2. The PDF documents are text based with embedded images that contain annotations e.g. maps, annotated satellite images, diagrams and photographs. I have a rudimentary extraction pipeline using pymupdf and gemma 27b to handle the images (gemma generates a text description of the image and extracts text). Any advice on a better way to prepare the PDF data would be appreciated. 3. Not particularly. It would ideally be very robust to detecting objects "out of distribution" e.g. ships being transported across land, or planes where they very rarely occur. 4. Zero shot would be excellent, but I'd be happy with few-shot if zero shot proves unreliable. The base use case still needs to be high probability of detection for common object types e.g. all ships and aircraft. 5. Yes - very interested.

---

ChatGPT Replied:
----------------

Thanks for the thorough answers. I’ll now investigate the best possible pipeline for state-of-the-art object detection and classification of small targets in very large geospatial satellite imagery, leveraging:

*   Foundation models (vision-language and vision transformers) for zero-shot and few-shot detection.
    
*   Scalable embedding strategies to allow flexible querying of stored satellite content.
    
*   Efficient use of your 128 A100 GPU setup.
    
*   Integration of multi-modal data from your domain-rich PDFs.
    
*   Approaches to maximize learning from 30TB of mostly unlabeled data and a limited set of 30,000 labeled objects.
    

I’ll also explore optimal strategies for pretraining or fine-tuning using your PDF content, as well as self-supervised or contrastive learning strategies for embedding generation.

I’ll get back to you with a full research plan and recommended tools, frameworks, and workflows shortly.

Research completed in 10m · 22 sources · 44 searches

Advanced Object Detection and Classification in Satellite Imagery Using Foundation Models

State-of-the-Art Small Object Detection in Gigapixel Satellite Images
=====================================================================

Introduction and Challenges
---------------------------

Detecting small, rare objects in **very large satellite images** (e.g. 50,000×100,000 pixels) is extremely challenging. Traditional supervised methods (e.g. fine-tuning an ImageNet-pretrained ResNet+Mask R-CNN) struggle to generalize beyond a fixed set of classes and require constant addition of new training data. Key challenges include:

*   **Gigapixel Image Size:** A single image may be billions of pixels, far exceeding typical CNN/transformer input sizes. Processing such imagery requires tiling or multi-scale strategies.
    
*   **Small Object Scale:** Targets like vehicles or buildings occupy only a few pixels to a few dozen pixels, making them hard to detect against complex backgrounds. They can be easily missed or confused with clutter.
    
*   **Rare Object Occurrence & Class Imbalance:** Some object types appear in only 1 out of 100 or 1000 images. Even common objects (cars, planes, ships) occupy a tiny fraction of pixels in an image. This sparsity means an **overwhelming background** dominates training, and models can quickly bias toward “no object” unless guided well.
    
*   **Limited Labeled Data:** You only have ~100 fully annotated images (~30,000 object instances) – not nearly enough to cover the variability of all object types and conditions.
    
*   **Abundant Unlabeled Data:** Conversely, you possess ~30 TB of unlabeled imagery. Leveraging this vast data via self-supervised or semi-supervised learning is crucial for success[arxiv.org](https://arxiv.org/html/2408.03464v1#:~:text=computational%20resources%2C%20and%20improved%20model,object%20detection%2C%20and%20other%20applications). Recent surveys note that _self-supervised pre-training (contrastive learning, masked autoencoders, etc.) significantly boosts remote sensing object detection performance and robustness_[arxiv.org](https://arxiv.org/html/2408.03464v1#:~:text=finds%20that%20pre,object%20detection%2C%20and%20other%20applications).
    
*   **Geo-Temporal Metadata:** Each image comes with geolocation, acquisition time, and sensor parameters. These could provide context (e.g. certain objects only in specific regions or times), though such metadata is rarely used in current detection models. Incorporating it without overfitting remains an open question.
    
*   **Evolving or Open Set of Target Classes:** You ultimately want the flexibility to detect **new object categories on the fly** – e.g. find “buildings that look like _this_” or a rare object described in text – without having to retrain a dedicated detector for each new class. This calls for an _open-vocabulary_ or _post-hoc query_ approach rather than a closed-set classifier.
    

In summary, the task demands a solution that can **learn from limited labels + abundant unlabeled data**, handle **gigapixel images and tiny objects**, exploit **textual domain knowledge**, and allow **flexible, future class definitions** (zero-shot or few-shot detection). Below we outline a comprehensive strategy leveraging **foundation models** and modern techniques to meet these requirements.

Leveraging Unlabeled Images with Foundation Models
--------------------------------------------------

Given the massive unlabeled image collection (30 TB), a top priority is to train or fine-tune a **foundation model** on this data. Foundation models – large-scale pretrained models – have revolutionized remote sensing by providing general features that transfer well to tasks like detection[arxiv.org](https://arxiv.org/html/2408.03464v1#:~:text=sensing%20has%20been%20significantly%20enhanced,achieved%20by%20those%20foundation%20models)[arxiv.org](https://arxiv.org/html/2408.03464v1#:~:text=computational%20resources%2C%20and%20improved%20model,object%20detection%2C%20and%20other%20applications). We propose:

*   **Self-Supervised Pretraining:** Use methods such as contrastive learning or masked image modeling on countless image _tiles_ cropped from your 30 TB archive. For example, approaches like **SimCLR/MoCo/SwAV** (contrastive) or **MAE/BEiT** (masked autoencoders) can learn robust feature representations without labels. In remote sensing, self-supervised pretraining (e.g. Seasonal Contrast (SeCo), DINO, SatMAE) has shown big gains in detection performance[arxiv.org](https://arxiv.org/html/2408.03464v1#:~:text=computational%20resources%2C%20and%20improved%20model,object%20detection%2C%20and%20other%20applications)[arxiv.org](https://arxiv.org/html/2408.03464v1#:~:text=On%20the%20DOTA%20dataset%2C%20RVSA,for%20specific%20remote%20sensing%20applications). A survey reports foundation models achieving **80%+ mAP** on benchmarks like DOTA (large-scale object detection dataset) after self-supervised pretraining[arxiv.org](https://arxiv.org/html/2408.03464v1#:~:text=On%20the%20DOTA%20dataset%2C%20RVSA,the%20best%20possible%20performance%20for) – far better than training from scratch. The learned backbone will serve as a strong feature extractor for small objects.
    
*   **Transformer Backbones:** Modern vision transformers (ViT, Swin Transformer) or hybrid CNN-Transformers are preferred as they handle long-range context well. Transformers have proven effective in remote sensing by modeling large receptive fields[arxiv.org](https://arxiv.org/html/2408.03464v1#:~:text=Transformers%2C%20originally%20designed%20for%20natural,proposed%20by%20Vaswani%20et%20al). Many recent remote-sensing foundation models use ViT or Swin backbones[arxiv.org](https://arxiv.org/html/2408.03464v1#:~:text=ResNet,B)[arxiv.org](https://arxiv.org/html/2408.03464v1#:~:text=%5B40%20%5D%20Levir,B%20SSL%2025.6M%2F87.8M). With 128 A100 GPUs, you can afford to train a hefty model (even ViT-Large or Swin-L) on millions of image patches.
    
*   **Tiling and Multi-Scale Training:** Since you cannot feed a 50k×100k image at full resolution into a model, implement a tiling strategy. For pretraining, randomly sample smaller sub-images (e.g. 512×512 or 1024×1024 pixels) across the dataset. Use aggressive data augmentation (random crops, rotations, etc.) to expose the model to varied scales and orientations of objects. You might also incorporate a multi-scale pyramid in the model (like an FPN or ViT with patch sizes) so that objects of different sizes can be detected.
    
*   **Geospatial Metadata Integration:** While pretraining, one idea is to concatenate metadata (location coordinates, timestamp, sensor ID) into the input. For instance, encode lat/long as sine/cosine features or a learned embedding, and append it to the image token sequence or into the model’s late layers. This could allow the model to learn correlations like _ships more likely near coasts_. However, this should be done carefully – the model must still detect _ships anywhere_ if they appear out-of-distribution. There is little literature on using geo-tags in detectors; most foundation models stick to imagery alone. One notable example is **GeoKR** which injected geographic knowledge into ResNet features[arxiv.org](https://arxiv.org/html/2408.03464v1#:~:text=Geographic%20Coverage%20Visual%20Encoder%20Pretrain,5M%20%202022%20Jul), but generally the gains from metadata are not well-explored. You can treat this as an experimental addition: perhaps start without it, then compare if adding a “metadata branch” improves precision on rare cases.
    

By the end of this stage, you would have a **strong, domain-specific visual backbone** (a pretrained transformer) that knows how to represent satellite imagery features. This backbone can be used for downstream detection and should be much more robust than ImageNet features for your task. For example, a foundation model “RVSA” (ResNet+ViT trained on remote images) achieved **81.2% mAP** on DOTA detection[arxiv.org](https://arxiv.org/html/2408.03464v1#:~:text=On%20the%20DOTA%20dataset%2C%20RVSA,the%20best%20possible%20performance%20for), whereas earlier methods without such pretraining were substantially lower. The rich feature representations from self-supervised learning will help discriminate tiny objects from clutter.

Utilizing Domain Knowledge from PDF Documents
---------------------------------------------

You have thousands of PDF documents containing annotated image chips, maps, and text descriptions – a treasure trove of _domain-specific knowledge_. We should **harness the multimodal (image + text) information** in these PDFs to guide the model. Possible steps:

*   **Image-Text Pair Extraction:** Develop a pipeline to extract each figure or image from the PDFs along with any corresponding caption or descriptive text in the document. Your current approach with PyMuPDF + Google Gemma-27B (multimodal model) is a good start. To improve it, ensure you capture: (a) the raw image (satellite chip or diagram), (b) any caption below the image, and (c) any call-outs or labels on the image (via OCR if text is embedded in the image). It might help to use a PDF-specific tool that preserves layout, so you can associate captions like “Figure 3: \[description\]…” with the right image. If the PDFs have annotated maps (e.g. red circles around objects), consider if those annotations can be parsed (some PDFs might have vector drawing layers). If not, you could even apply an object detector to the extracted figure to find the marked objects (since those figures are smaller, this is feasible).
    
*   **Build a Remote Sensing “CLIP” dataset:** Using the extracted image-text pairs, you can create a dataset for **vision-language pretraining**. Each image chip from the PDF, paired with its description or the surrounding text, becomes a training sample. Even if you only get on the order of 10k–100k such pairs, they carry highly relevant semantics (e.g. “a missile launcher hiding under camouflage netting in a desert”). This can be used to _fine-tune a CLIP model_ or similar on your domain. Recent work shows that adapting CLIP to remote sensing yields huge benefits for zero-shot recognition. For instance, **RS-CLIP**, a CLIP model trained on remote sensing image-text data, achieved up to **95% accuracy in zero-shot scene classification**, vastly outperforming the original CLIP on that domain[centaur.reading.ac.uk](https://centaur.reading.ac.uk/119820/#:~:text=CentAUR%20centaur,21%2C). In your case, a domain-tuned CLIP will learn the language of your PDFs – linking phrases to satellite imagery patterns.
    
*   **Vision-Language Model Integration:** Once you have a domain CLIP (or a similar vision-language model), incorporate it into the detection pipeline (detailed next). The simplest use-case is **search and retrieval**: you could embed all your imagery (or candidate object patches) into CLIP’s visual embedding space, and embed a text query (e.g. _“small building with a dome roof”_) into the text space – the model can then find image regions with high similarity to the query. This directly enables _zero-shot object search_ over your image archive. Indeed, open-vocabulary detection methods often rely on CLIP embeddings to classify proposals by comparing image vs. text features in a shared space[ar5iv.labs.arxiv.org](https://ar5iv.labs.arxiv.org/html/2311.11646#:~:text=on%20massive%20image,transfers%20knowledge%20from%20a%20pre). Keep in mind the finding by Bou _et al._ (2024) that a generic CLIP may lack fine-grained remote sensing vocabulary, whereas purely visual features like DINOv2 performed better on satellite object detection[arxiv.org](https://arxiv.org/abs/2403.05381#:~:text=containing%20challenging%20and%20rare%20objects,datasets%2C%20despite%20minimal%20training%20parameters). **Solution:** use the PDF text to teach the model domain-specific vocabulary (class names and traits). By training CLIP on your data (or at least fine-tuning it), the text encoder will learn terms like _“SAM site”_, _“mobile launcher”_, _“dry dock”_, etc., which don’t appear often in ordinary CLIP training. This will narrow the gap so that text embeddings truly reflect your target objects.
    
*   **Multimodal Prototype Learning:** Another way to use text is as _“prototypes”_ for classes. Research on few-shot detection in remote sensing suggests combining **textual descriptions with visual exemplars** yields the best results[mdpi.com](https://www.mdpi.com/2072-4292/16/24/4693#:~:text=detection%2C%20constructing%20class%20prototypes%20from,both%20worlds%2C%20we%20propose%20a)[mdpi.com](https://www.mdpi.com/2072-4292/16/24/4693#:~:text=textual%20class%20names%20are%20processed,modal). Intuitively, a text embedding (from a language model like BERT or CLIP’s text tower) carries high-level semantic knowledge (e.g. a “ship” is a watercraft, usually in water, etc.), while a visual embedding of a few example chips carries specific appearance details. Liu _et al._ (2024) fuse textual and visual prototypes to create multi-modal class representations, and report up to **+8.7% AP50 improvement** in few-shot object detection performance[mdpi.com](https://www.mdpi.com/2072-4292/16/24/4693#:~:text=prototype%20aggregating%20module%20to%20integrate,remote%20sensing%20scenarios%20remains%20unexplored)[mdpi.com](https://www.mdpi.com/2072-4292/16/24/4693#:~:text=Figure%202,art%20approaches). We can adopt a similar idea: for any new target class, define it by a textual description _and_ (if available) a few image examples. The detector’s classifier can then be a _matching function_ that checks a region’s feature against these multi-modal class prototypes. This approach is more flexible than a fixed linear classifier and naturally extends to new classes without retraining the entire model.
    

In summary, the PDF-derived data allows us to imbue the model with **contextual and semantic knowledge** beyond what raw pixels provide. It’s like having an expert guide the model: the text can describe _how to recognize a rare object_, and the image chips provide exemplars. We will leverage this in building an **open-vocabulary detection system** below.

Open-Vocabulary Detection and Classification
--------------------------------------------

Instead of training a conventional detector locked to N predefined classes, we recommend an **open-vocabulary object detection (OVD)** approach. This means the detector can potentially detect _any object category_, given a suitable description or example of that category – addressing your need to add new classes without reprocessing all images. There are a few components to achieve this:

*   **Region Proposal Network (RPN) or Segmenter:** First, we need to generate candidate regions in the large image that _might contain any object_. These proposals should be _class-agnostic_. Traditional detectors use either sliding-window anchors or learned RPNs that are trained on known objects. For open-vocab, we can still train an RPN on the base classes (like vehicles) just to get it to propose regions that _look “objecty”_ in general. Alternatively, we can use a powerful pretrained segmenter like **SAM (Segment Anything Model)** to propose masks for **all objects** in an image. SAM, when given an image, can generate ~everything that looks like a discrete object or part in that scene. It has been shown to work on satellite images as well. The idea would be: run SAM (or an RPN) on each image tile to generate a set of candidate object bounding boxes or masks (this can be done offline and stored if needed). This addresses finding the needles in the haystack (small objects in huge images) without yet knowing what they are.
    
*   **Feature Extraction for Proposals:** For each candidate region, extract a feature descriptor using our **foundation backbone** (the one pretrained on 30TB of imagery). This could be, for example, a ViT feature vector from that region (or a RoI pooled feature from a CNN). Because the backbone was trained on diverse data, these features will encode the object’s visual appearance in a rich way.
    
*   **Open-Vocabulary Classification:** Now, instead of a fixed classifier head, classify a region by comparing its feature to **embeddings of class names or descriptions**. This is where the domain **CLIP** model (or a text encoder) comes in. We obtain a text embedding for each class of interest (e.g. the word “aircraft”, or a phrase “small propeller airplane”). The region is assigned the class whose text embedding is most similar to the region’s visual embedding (via cosine similarity)[ar5iv.labs.arxiv.org](https://ar5iv.labs.arxiv.org/html/2311.11646#:~:text=on%20massive%20image,transfers%20knowledge%20from%20a%20pre). This effectively lets us have _thousands of potential classes_ without training a separate neuron for each – we just compute their text embeddings. If you have example images of a new class rather than a name, you can also embed those through the vision encoder and use them as “visual class prototypes” in the same embedding space. Recent open-vocab detectors follow this paradigm. For example, **ViLD (Viision-Language Detection)** by Google used a CLIP image encoder to embed region proposals and a CLIP text encoder for class names, matching them in a shared space[ar5iv.labs.arxiv.org](https://ar5iv.labs.arxiv.org/html/2311.11646#:~:text=on%20massive%20image,transfers%20knowledge%20from%20a%20pre). Microsoft’s **GLIP** went further by _jointly training_ a detector on detection data _plus_ image-caption pairs, so it learned to align detection with text descriptions directly[ar5iv.labs.arxiv.org](https://ar5iv.labs.arxiv.org/html/2311.11646#:~:text=vocabularies%20than%20conventional%20nature%20image,class%20similarity%20and). In your case, we can fine-tune such a model on your data (using your 100 labeled images for supervision on base classes like ships/planes, while also exposing it to image-text pairs from the PDFs to teach the alignment). The result will be a detector that can take a **text query at inference (“find building with a dome”)** and output boxes where such objects are present.
    
*   **Student-Teacher Self-Training:** Since you have so much unlabeled data, we should also exploit _pseudo-labeling_ to expand to novel classes. A cutting-edge approach, **CastDet (ECCV 2024)**, did exactly this for aerial images: it used a CLIP-based teacher to propose and label new objects in unlabeled images, and a student detector to learn from those pseudo-labels[arxiv.org](https://arxiv.org/abs/2311.11646#:~:text=performance%20of%20OVD%20greatly%20relies,We%20conduct%20extensive)[ar5iv.labs.arxiv.org](https://ar5iv.labs.arxiv.org/html/2311.11646#:~:text=pseudo,extensive%20experiments%20on%20multiple%20existing). Essentially, the CLIP model can score proposals for any class via the text embeddings – we take high-confidence detections on unlabeled images as new “labels” and retrain/refine the detector. This iterative self-training can discover **rare objects** that were not in the original 100 labeled images. CastDet reported a **large improvement (≈+21 mAP)** on novel object detection in aerial scenes by using a CLIP teacher in this way[arxiv.org](https://arxiv.org/abs/2311.11646#:~:text=up%20for%20the%20OVD%20task,available%20at%20this%20https%20URL). You could employ a similar strategy: use your domain-CLIP as a teacher to tag occurrences of, say, a **new vehicle type** across the unlabeled dataset, then incorporate those as additional training data. Over time, the model’s vocabulary of detectable objects grows, without explicit human labeling.
    

_Open-vocabulary detection concept: Instead of fixed-class prediction, the detector compares proposal features with text or image embeddings. (Left) A traditional CLIP-based OVD uses CLIP as a frozen teacher to classify proposals; (Right) a hybrid approach injects CLIP knowledge into a student–teacher detection model for learning new classes[ar5iv.labs.arxiv.org](https://ar5iv.labs.arxiv.org/html/2311.11646#:~:text=Image%3A%20Refer%20to%20caption%20Figure,an%20external%20knowledge%20source%20to)._

The above figure illustrates how we can integrate a CLIP-like model into the training process. In our implementation, the “teacher” could be the domain-adapted CLIP (trained on PDF data) that guides the detector to high-quality proposals and provides semantic labels for them[ar5iv.labs.arxiv.org](https://ar5iv.labs.arxiv.org/html/2311.11646#:~:text=pseudo,extensive%20experiments%20on%20multiple%20existing). The “student” is our main detector that we train end-to-end. This way, the **broad knowledge in CLIP (about many object types)** is distilled into the detector. By the end, you should have a model that detects **common objects with high precision** (thanks to supervised training on those 30k labels) and can also **localize unseen objects** given a description or few examples (thanks to the open-vocab design and CLIP’s knowledge).

A few additional notes on making this work:

*   **Thresholding and Calibration:** In zero-shot settings, false positives can be an issue (the model might see _“objects”_ everywhere for a vague query). You will need to calibrate similarity score thresholds for detection. One trick is to always include a “background” or “none of the above” category embedding in the comparison – for example, an embedding of the word “background” or an average of many random image embeddings – to represent no-object. The detector should only output a box if its similarity to a real class exceeds similarity to “background” by a margin. This was used in some OVD methods to reduce false alarms.
    
*   **Few-Shot Fine-tuning:** If zero-shot detection of a new class is not sufficiently accurate, the framework easily supports a _few-shot fine-tune_. Because the backbone is strong and the classifier is based on embeddings, training on say 10–20 labeled examples of the new class can quickly adjust the image embedding space to that class. The few-shot learning could be done by updating prototype vectors or by fine-tuning the detector’s head slightly – far less effort than adding a class in a traditional detector (which would normally require full retraining).
    
*   **Output Formats:** Since you originally used Mask R-CNN, you might want segmentation masks. You can indeed get masks by applying something like **Mask2Former** or using SAM on the detected boxes. But primarily, the pipeline gives bounding boxes. If masks are needed, one idea is to incorporate SAM in the inference loop: for each detected box, run SAM with a prompt point inside that box to get a precise segmentation. Given SAM is zero-shot for segmentation, it pairs well with an open-vocab detector which is zero-shot for classification.
    
*   **Real-Time Querying vs Pre-indexing:** You wondered if you could _“store image content (embeddings) and define objects of interest after the fact”_ instead of reprocessing images for each new query. In practice, a hybrid approach is possible. For example, you could **precompute and store the visual embeddings of all proposals** in each large image (perhaps store the feature vectors for all SAM-proposed regions, or for a grid of patches). Then, later on, when you have a new object query, you **only need to embed the query text or image and compare to the stored features** – essentially a nearest-neighbor search in the embedding space. This is feasible using vector databases or even simple indexing if the data isn’t too huge. Storing embeddings for _every possible patch_ of 30TB is not feasible, but storing ~hundreds of proposals per image is. For instance, if SAM or RPN gives 500 candidate regions in an image, that’s 500 feature vectors (say 512-D each) per image – which across, say, 100k images is 50 million vectors, manageable with approximate nearest neighbor search. This way, you can avoid recomputing deep features on the entire archive for each new object search; you only do a fast embedding lookup. Many image search systems adopt this paradigm. As long as your feature space is well-aligned with semantics, this will allow **post-hoc querying** of the archive for arbitrary objects.
    

By designing the detector in an open-vocabulary manner, we essentially future-proof it: adding a new target class means _writing a new prompt or providing an example_, rather than labelling thousands of images and retraining a monolithic model. This addresses your concern that with previous methods, _“adding classes is difficult and requires reprocessing old images”_. With our approach, old images have already been processed into embeddings; finding a new class in them is a matter of querying those embeddings, not rerunning a full detection model from scratch.

Robustness to Unusual and Rare Instances
----------------------------------------

Another important goal is to ensure the system is **robust to out-of-distribution occurrences** – e.g. a ship on land or an airplane in an odd location. Our approach tackles this on multiple fronts:

*   **Diverse Training Data:** The self-supervised pretraining on 30TB of imagery will expose the model to countless environments and contexts, more than any supervised set could. This broad exposure tends to make the features more context-agnostic – the model learns to focus on objects themselves, not just typical context. In fact, foundation models are praised for their _robustness and generalization_ in remote sensing tasks[arxiv.org](https://arxiv.org/html/2408.03464v1#:~:text=trends%20and%20the%20significant%20advancements,object%20detection%2C%20and%20other%20applications). We will also augment the supervised fine-tuning with any “odd” examples we can find (perhaps your PDFs or archives contain a few examples of ships inland, etc.). By explicitly including such cases in training (even if only a few), we can teach the model that, say, _“a ship is a ship, even if on land”_.
    
*   **Open-Vocab Semantics:** Using semantic descriptors for detection means the model relies on the object’s visual and textual definition, rather than the surrounding pixels. For example, if we query “ship”, the detector will look for regions that _look like a ship_ (long, hull, etc.) regardless of background. In contrast, a conventional detector trained only on ships-in-water might have implicitly learned that water context = ship. Our approach is less prone to that trap because the “ship” prototype is defined by either a text description (“watercraft with deck and hull”) or by diverse examples (some could be ships in docks, ships on land from scrapyards, etc.). This should inherently improve robustness to odd contexts.
    
*   **Few-Shot Updates:** If a truly novel scenario pops up (say, a new camouflage or a new platform we’ve never seen), the system can quickly adapt by few-shot learning. Because it’s designed to ingest new classes or variants easily, you could feed a handful of examples of the new appearance and update the prototypes. This agility means the model won’t remain brittle in the face of new distribution shifts.
    
*   **Evaluation & Iteration:** It’s worth setting aside some of your data as a _validation set specifically for unusual cases_. Monitor the detector’s performance on these (e.g. a plane in a building, a ship on land). If it struggles, you might generate synthetic data to help. For instance, you could paste object chips into uncommon contexts as augmented training data (copy a ship onto a land background). With diffusion models or image editing, one can even create semi-realistic composites. This kind of targeted augmentation could be a short-term hack to improve robustness.
    

Ultimately, a combination of _massive pretraining_, _semantic open-class reasoning_, and _targeted fine-tuning_ should yield a detector that doesn’t blink when encountering weird situations – it will identify the object for what it is, not where it is.

Training at Scale and Timeline
------------------------------

Given your 128×80GB GPU setup and ~2-week timeline for initial results, a phased training plan is realistic:

1.  **Week 1:** Focus on self-supervised pretraining of the backbone on unlabeled images. Use a distributed training framework (PyTorch DDP or Lightning, etc.) to spread batches across all GPUs. With 128 A100s, you can process enormous batches quickly – perhaps in the order of 1e5 images per epoch. In one week, you could conceivably run a few epochs over a significant subset of the 30TB (depending on I/O). Choose an SSL method that is known to converge fast; _DINOv2_ by Meta, for example, is a recipe that produced a robust ViT by training on 1 billion images (though that took more compute than we have, but we can aim for a smaller scale). Monitor the loss and ensure the model is indeed learning features (e.g. visualize some attention maps or do a quick linear probe to verify it's not random).
    
2.  **Week 2:** Set up the open-vocabulary detection fine-tuning. This stage uses your ~100 labeled images (with 30k labeled objects) as supervised data for the detection task _and_ possibly uses pseudo-labels and the image-text data. A possible curriculum:
    
    *   Fine-tune the pretrained backbone with a detection head (say a DETR or Faster R-CNN style head) on the **base classes** (ships, planes, etc.) for which you have labels. Also include a background class. Use multi-scale training if possible (e.g. random image rescaling) to help detect various sizes. Because the backbone is already good, this fine-tuning can reach high mAP on these classes relatively quickly (a few days of training at most).
        
    *   Next, integrate the **CLIP-based classification**. Freeze or remove the old softmax classifier, and instead attach the embedding-based classifier. You can either freeze the backbone at this point (to preserve its learned features) and just train a small adaptation layer, or continue fine-tuning lightly. Train the detector to output embeddings for proposals that match the ground-truth class embeddings. For example, for each GT box of class “ship”, treat the text embedding of “ship” as a target, and train the proposal feature to be close to it (this is like knowledge distillation from class name to proposal). Microsoft’s **Detic** (Zhou et al. 2022) did something similar by using hundreds of class names as weak supervision for detection. This step will align the detector’s feature space with the text space.
        
    *   Meanwhile, use the **PDF image-text pairs** to train the text encoder (if not done in week 1). Possibly do this in parallel on separate GPUs. Once the text encoder is ready, you can use it in the above step. You could also do joint training where detection and image-text contrastive learning happen together (a multi-task learning approach), though that is more complex.
        
    *   If time permits, run a round of **pseudo-labeling**: apply the emerging open-vocab detector to a portion of the unlabeled 30TB images to see if it discovers new objects. For example, have it output any high-confidence detections for the prompt set {“ship”, “airplane”, “vehicle”, “building”} on those images. You might find that it detects some ships or planes in new places. Add these pseudo-labeled instances to the training set (with appropriate filtering for precision), and continue training. This can gradually expand the model’s coverage.
        

Given two weeks, you might not fully iterate over all steps, but you can get an initial model out and tested. The beauty of this approach is that it is **incremental** – you can keep improving the model with more unlabeled data and more classes over time, without starting over. The heavy lift is mostly up front (pretraining the foundation). After that, each new class or new dataset can be folded in with relatively little compute.

Conclusion and Future Directions
--------------------------------

By combining a **foundation model** for vision (robust features from massive unlabeled data) with a **vision-language approach** (to leverage domain knowledge and enable open-set queries), you will have a cutting-edge object detection system for satellite images. This system should:

*   Achieve high detection accuracy on known target classes (e.g. all ships and aircraft) by leveraging both labeled data and abundant unlabeled imagery (via self-supervised learning and semi-supervised fine-tuning). Expect a significant boost in detection performance compared to your older ResNet50-based model – foundation models in remote sensing have reported substantial mAP gains on benchmarks[arxiv.org](https://arxiv.org/html/2408.03464v1#:~:text=On%20the%20DOTA%20dataset%2C%20RVSA,for%20specific%20remote%20sensing%20applications).
    
*   **Generalize to new object categories** with zero or few examples, thanks to the open-vocabulary design. You’ll be able to search historical imagery for objects of interest defined by a text description or a prototype image – without retraining a specialized detector each time. This addresses the scalability issue of adding new classes. The detector essentially becomes a general _“query-able”_ index of your imagery.
    
*   **Adapt to unusual scenarios** better than before. Through broad training and semantic understanding, the model is less likely to be confused by context. A plane in an unexpected location should still trigger a _“plane”_ detection because the visual features match, and the model’s semantic priors know that planes **can** be on roads or museums, etc. Moreover, any remaining blind spots can be patched by feeding a few examples to the model (few-shot learning), rather than requiring a full new dataset.
    
*   Utilize **multimodal data** (imagery + text) in a unified way. The PDF documents’ information will enrich the model’s knowledge – essentially functioning as a form of transfer learning from human domain experts. This is relatively novel in automated object detection, so you’d be at the forefront of exploiting textual intelligence in a vision model. Research indicates that combining textual and visual cues yields superior performance in remote sensing tasks[mdpi.com](https://www.mdpi.com/2072-4292/16/24/4693#:~:text=prototype%20aggregating%20module%20to%20integrate,remote%20sensing%20scenarios%20remains%20unexplored)[mdpi.com](https://www.mdpi.com/2072-4292/16/24/4693#:~:text=extraction%20capabilities%20of%20prototypes,generalization%20capabilities%2C%20as%20the%20text), which bodes well for your application.
    

Finally, ensure to set up a strong evaluation pipeline. Given the open-ended nature, you’ll want to evaluate not just on traditional metrics (mAP on a fixed set of classes) but also test the system’s ability to retrieve novel targets. You could simulate “zero-shot” tests by holding out some object classes during training and then asking the model to find them via text prompts, measuring how well it recovers them. This will give a sense of its true discovery capabilities. Keep an eye on failure modes: are there systematic false positives (e.g. confusing ships vs. buildings in some conditions)? Are there biases from the PDF data (e.g. the model only recognizes an object if described in a certain way)? Use such insights to refine the training – the modular design (separating vision and language components) makes it easier to tweak one part without breaking everything.

In summary, the path forward is to **train a powerful general-purpose feature extractor on your satellite imagery**, and **marry it with a flexible detection head driven by both visual and linguistic knowledge**. This approach is state-of-the-art in both research and practice for scenarios exactly like yours. With your abundant data and compute, you are well-positioned to push the state of the art in small object detection in large-scale satellite images. Good luck, and happy detecting!

**Sources:** Remote sensing foundation model survey[arxiv.org](https://arxiv.org/html/2408.03464v1#:~:text=sensing%20has%20been%20significantly%20enhanced,achieved%20by%20those%20foundation%20models)[arxiv.org](https://arxiv.org/html/2408.03464v1#:~:text=computational%20resources%2C%20and%20improved%20model,object%20detection%2C%20and%20other%20applications); Performance of foundation models on object detection[arxiv.org](https://arxiv.org/html/2408.03464v1#:~:text=On%20the%20DOTA%20dataset%2C%20RVSA,the%20best%20possible%20performance%20for)[arxiv.org](https://arxiv.org/html/2408.03464v1#:~:text=On%20the%20DOTA%20dataset%2C%20RVSA,for%20specific%20remote%20sensing%20applications); Open-vocabulary detection concept and CLIP integration[ar5iv.labs.arxiv.org](https://ar5iv.labs.arxiv.org/html/2311.11646#:~:text=on%20massive%20image,transfers%20knowledge%20from%20a%20pre)[ar5iv.labs.arxiv.org](https://ar5iv.labs.arxiv.org/html/2311.11646#:~:text=ViLD%5B9%20%5D%2C%20OV,to%20a%20low%20recall%20for); Few-shot detection with multi-modal prototypes[mdpi.com](https://www.mdpi.com/2072-4292/16/24/4693/pdf?version=1734514531#:~:text=prototype,and%20spatial%20details%20from%20visual)[mdpi.com](https://www.mdpi.com/2072-4292/16/24/4693#:~:text=textual%20class%20names%20are%20processed,modal); CastDet open-vocabulary aerial detection[arxiv.org](https://arxiv.org/abs/2311.11646#:~:text=performance%20of%20OVD%20greatly%20relies,We%20conduct%20extensive)[arxiv.org](https://arxiv.org/abs/2311.11646#:~:text=up%20for%20the%20OVD%20task,available%20at%20this%20https%20URL); CLIP vs. visual features in satellite imagery[arxiv.org](https://arxiv.org/abs/2403.05381#:~:text=containing%20challenging%20and%20rare%20objects,datasets%2C%20despite%20minimal%20training%20parameters).

![](https://www.google.com/s2/favicons?domain=https://www.mdpi.com&sz=32)

![](https://www.google.com/s2/favicons?domain=https://ar5iv.labs.arxiv.org&sz=32)

![](https://www.google.com/s2/favicons?domain=https://centaur.reading.ac.uk&sz=32)

![](https://www.google.com/s2/favicons?domain=https://arxiv.org&sz=32)

Sources

---

ChatGPT Replied:
----------------

Advanced Object Detection and Classification in Satellite Imagery Using Foundation Models

State-of-the-Art Small Object Detection in Gigapixel Satellite Images
=====================================================================

Introduction and Challenges
---------------------------

Detecting small, rare objects in **very large satellite images** (e.g. 50,000×100,000 pixels) is extremely challenging. Traditional supervised methods (e.g. fine-tuning an ImageNet-pretrained ResNet+Mask R-CNN) struggle to generalize beyond a fixed set of classes and require constant addition of new training data. Key challenges include:

*   **Gigapixel Image Size:** A single image may be billions of pixels, far exceeding typical CNN/transformer input sizes. Processing such imagery requires tiling or multi-scale strategies.
    
*   **Small Object Scale:** Targets like vehicles or buildings occupy only a few pixels to a few dozen pixels, making them hard to detect against complex backgrounds. They can be easily missed or confused with clutter.
    
*   **Rare Object Occurrence & Class Imbalance:** Some object types appear in only 1 out of 100 or 1000 images. Even common objects (cars, planes, ships) occupy a tiny fraction of pixels in an image. This sparsity means an **overwhelming background** dominates training, and models can quickly bias toward “no object” unless guided well.
    
*   **Limited Labeled Data:** You only have ~100 fully annotated images (~30,000 object instances) – not nearly enough to cover the variability of all object types and conditions.
    
*   **Abundant Unlabeled Data:** Conversely, you possess ~30 TB of unlabeled imagery. Leveraging this vast data via self-supervised or semi-supervised learning is crucial for success[arxiv.org](https://arxiv.org/html/2408.03464v1#:~:text=computational%20resources%2C%20and%20improved%20model,object%20detection%2C%20and%20other%20applications). Recent surveys note that _self-supervised pre-training (contrastive learning, masked autoencoders, etc.) significantly boosts remote sensing object detection performance and robustness_[arxiv.org](https://arxiv.org/html/2408.03464v1#:~:text=finds%20that%20pre,object%20detection%2C%20and%20other%20applications).
    
*   **Geo-Temporal Metadata:** Each image comes with geolocation, acquisition time, and sensor parameters. These could provide context (e.g. certain objects only in specific regions or times), though such metadata is rarely used in current detection models. Incorporating it without overfitting remains an open question.
    
*   **Evolving or Open Set of Target Classes:** You ultimately want the flexibility to detect **new object categories on the fly** – e.g. find “buildings that look like _this_” or a rare object described in text – without having to retrain a dedicated detector for each new class. This calls for an _open-vocabulary_ or _post-hoc query_ approach rather than a closed-set classifier.
    

In summary, the task demands a solution that can **learn from limited labels + abundant unlabeled data**, handle **gigapixel images and tiny objects**, exploit **textual domain knowledge**, and allow **flexible, future class definitions** (zero-shot or few-shot detection). Below we outline a comprehensive strategy leveraging **foundation models** and modern techniques to meet these requirements.

Leveraging Unlabeled Images with Foundation Models
--------------------------------------------------

Given the massive unlabeled image collection (30 TB), a top priority is to train or fine-tune a **foundation model** on this data. Foundation models – large-scale pretrained models – have revolutionized remote sensing by providing general features that transfer well to tasks like detection[arxiv.org](https://arxiv.org/html/2408.03464v1#:~:text=sensing%20has%20been%20significantly%20enhanced,achieved%20by%20those%20foundation%20models)[arxiv.org](https://arxiv.org/html/2408.03464v1#:~:text=computational%20resources%2C%20and%20improved%20model,object%20detection%2C%20and%20other%20applications). We propose:

*   **Self-Supervised Pretraining:** Use methods such as contrastive learning or masked image modeling on countless image _tiles_ cropped from your 30 TB archive. For example, approaches like **SimCLR/MoCo/SwAV** (contrastive) or **MAE/BEiT** (masked autoencoders) can learn robust feature representations without labels. In remote sensing, self-supervised pretraining (e.g. Seasonal Contrast (SeCo), DINO, SatMAE) has shown big gains in detection performance[arxiv.org](https://arxiv.org/html/2408.03464v1#:~:text=computational%20resources%2C%20and%20improved%20model,object%20detection%2C%20and%20other%20applications)[arxiv.org](https://arxiv.org/html/2408.03464v1#:~:text=On%20the%20DOTA%20dataset%2C%20RVSA,for%20specific%20remote%20sensing%20applications). A survey reports foundation models achieving **80%+ mAP** on benchmarks like DOTA (large-scale object detection dataset) after self-supervised pretraining[arxiv.org](https://arxiv.org/html/2408.03464v1#:~:text=On%20the%20DOTA%20dataset%2C%20RVSA,the%20best%20possible%20performance%20for) – far better than training from scratch. The learned backbone will serve as a strong feature extractor for small objects.
    
*   **Transformer Backbones:** Modern vision transformers (ViT, Swin Transformer) or hybrid CNN-Transformers are preferred as they handle long-range context well. Transformers have proven effective in remote sensing by modeling large receptive fields[arxiv.org](https://arxiv.org/html/2408.03464v1#:~:text=Transformers%2C%20originally%20designed%20for%20natural,proposed%20by%20Vaswani%20et%20al). Many recent remote-sensing foundation models use ViT or Swin backbones[arxiv.org](https://arxiv.org/html/2408.03464v1#:~:text=ResNet,B)[arxiv.org](https://arxiv.org/html/2408.03464v1#:~:text=%5B40%20%5D%20Levir,B%20SSL%2025.6M%2F87.8M). With 128 A100 GPUs, you can afford to train a hefty model (even ViT-Large or Swin-L) on millions of image patches.
    
*   **Tiling and Multi-Scale Training:** Since you cannot feed a 50k×100k image at full resolution into a model, implement a tiling strategy. For pretraining, randomly sample smaller sub-images (e.g. 512×512 or 1024×1024 pixels) across the dataset. Use aggressive data augmentation (random crops, rotations, etc.) to expose the model to varied scales and orientations of objects. You might also incorporate a multi-scale pyramid in the model (like an FPN or ViT with patch sizes) so that objects of different sizes can be detected.
    
*   **Geospatial Metadata Integration:** While pretraining, one idea is to concatenate metadata (location coordinates, timestamp, sensor ID) into the input. For instance, encode lat/long as sine/cosine features or a learned embedding, and append it to the image token sequence or into the model’s late layers. This could allow the model to learn correlations like _ships more likely near coasts_. However, this should be done carefully – the model must still detect _ships anywhere_ if they appear out-of-distribution. There is little literature on using geo-tags in detectors; most foundation models stick to imagery alone. One notable example is **GeoKR** which injected geographic knowledge into ResNet features[arxiv.org](https://arxiv.org/html/2408.03464v1#:~:text=Geographic%20Coverage%20Visual%20Encoder%20Pretrain,5M%20%202022%20Jul), but generally the gains from metadata are not well-explored. You can treat this as an experimental addition: perhaps start without it, then compare if adding a “metadata branch” improves precision on rare cases.
    

By the end of this stage, you would have a **strong, domain-specific visual backbone** (a pretrained transformer) that knows how to represent satellite imagery features. This backbone can be used for downstream detection and should be much more robust than ImageNet features for your task. For example, a foundation model “RVSA” (ResNet+ViT trained on remote images) achieved **81.2% mAP** on DOTA detection[arxiv.org](https://arxiv.org/html/2408.03464v1#:~:text=On%20the%20DOTA%20dataset%2C%20RVSA,the%20best%20possible%20performance%20for), whereas earlier methods without such pretraining were substantially lower. The rich feature representations from self-supervised learning will help discriminate tiny objects from clutter.

Utilizing Domain Knowledge from PDF Documents
---------------------------------------------

You have thousands of PDF documents containing annotated image chips, maps, and text descriptions – a treasure trove of _domain-specific knowledge_. We should **harness the multimodal (image + text) information** in these PDFs to guide the model. Possible steps:

*   **Image-Text Pair Extraction:** Develop a pipeline to extract each figure or image from the PDFs along with any corresponding caption or descriptive text in the document. Your current approach with PyMuPDF + Google Gemma-27B (multimodal model) is a good start. To improve it, ensure you capture: (a) the raw image (satellite chip or diagram), (b) any caption below the image, and (c) any call-outs or labels on the image (via OCR if text is embedded in the image). It might help to use a PDF-specific tool that preserves layout, so you can associate captions like “Figure 3: \[description\]…” with the right image. If the PDFs have annotated maps (e.g. red circles around objects), consider if those annotations can be parsed (some PDFs might have vector drawing layers). If not, you could even apply an object detector to the extracted figure to find the marked objects (since those figures are smaller, this is feasible).
    
*   **Build a Remote Sensing “CLIP” dataset:** Using the extracted image-text pairs, you can create a dataset for **vision-language pretraining**. Each image chip from the PDF, paired with its description or the surrounding text, becomes a training sample. Even if you only get on the order of 10k–100k such pairs, they carry highly relevant semantics (e.g. “a missile launcher hiding under camouflage netting in a desert”). This can be used to _fine-tune a CLIP model_ or similar on your domain. Recent work shows that adapting CLIP to remote sensing yields huge benefits for zero-shot recognition. For instance, **RS-CLIP**, a CLIP model trained on remote sensing image-text data, achieved up to **95% accuracy in zero-shot scene classification**, vastly outperforming the original CLIP on that domain[centaur.reading.ac.uk](https://centaur.reading.ac.uk/119820/#:~:text=CentAUR%20centaur,21%2C). In your case, a domain-tuned CLIP will learn the language of your PDFs – linking phrases to satellite imagery patterns.
    
*   **Vision-Language Model Integration:** Once you have a domain CLIP (or a similar vision-language model), incorporate it into the detection pipeline (detailed next). The simplest use-case is **search and retrieval**: you could embed all your imagery (or candidate object patches) into CLIP’s visual embedding space, and embed a text query (e.g. _“small building with a dome roof”_) into the text space – the model can then find image regions with high similarity to the query. This directly enables _zero-shot object search_ over your image archive. Indeed, open-vocabulary detection methods often rely on CLIP embeddings to classify proposals by comparing image vs. text features in a shared space[ar5iv.labs.arxiv.org](https://ar5iv.labs.arxiv.org/html/2311.11646#:~:text=on%20massive%20image,transfers%20knowledge%20from%20a%20pre). Keep in mind the finding by Bou _et al._ (2024) that a generic CLIP may lack fine-grained remote sensing vocabulary, whereas purely visual features like DINOv2 performed better on satellite object detection[arxiv.org](https://arxiv.org/abs/2403.05381#:~:text=containing%20challenging%20and%20rare%20objects,datasets%2C%20despite%20minimal%20training%20parameters). **Solution:** use the PDF text to teach the model domain-specific vocabulary (class names and traits). By training CLIP on your data (or at least fine-tuning it), the text encoder will learn terms like _“SAM site”_, _“mobile launcher”_, _“dry dock”_, etc., which don’t appear often in ordinary CLIP training. This will narrow the gap so that text embeddings truly reflect your target objects.
    
*   **Multimodal Prototype Learning:** Another way to use text is as _“prototypes”_ for classes. Research on few-shot detection in remote sensing suggests combining **textual descriptions with visual exemplars** yields the best results[mdpi.com](https://www.mdpi.com/2072-4292/16/24/4693#:~:text=detection%2C%20constructing%20class%20prototypes%20from,both%20worlds%2C%20we%20propose%20a)[mdpi.com](https://www.mdpi.com/2072-4292/16/24/4693#:~:text=textual%20class%20names%20are%20processed,modal). Intuitively, a text embedding (from a language model like BERT or CLIP’s text tower) carries high-level semantic knowledge (e.g. a “ship” is a watercraft, usually in water, etc.), while a visual embedding of a few example chips carries specific appearance details. Liu _et al._ (2024) fuse textual and visual prototypes to create multi-modal class representations, and report up to **+8.7% AP50 improvement** in few-shot object detection performance[mdpi.com](https://www.mdpi.com/2072-4292/16/24/4693#:~:text=prototype%20aggregating%20module%20to%20integrate,remote%20sensing%20scenarios%20remains%20unexplored)[mdpi.com](https://www.mdpi.com/2072-4292/16/24/4693#:~:text=Figure%202,art%20approaches). We can adopt a similar idea: for any new target class, define it by a textual description _and_ (if available) a few image examples. The detector’s classifier can then be a _matching function_ that checks a region’s feature against these multi-modal class prototypes. This approach is more flexible than a fixed linear classifier and naturally extends to new classes without retraining the entire model.
    

In summary, the PDF-derived data allows us to imbue the model with **contextual and semantic knowledge** beyond what raw pixels provide. It’s like having an expert guide the model: the text can describe _how to recognize a rare object_, and the image chips provide exemplars. We will leverage this in building an **open-vocabulary detection system** below.

Open-Vocabulary Detection and Classification
--------------------------------------------

Instead of training a conventional detector locked to N predefined classes, we recommend an **open-vocabulary object detection (OVD)** approach. This means the detector can potentially detect _any object category_, given a suitable description or example of that category – addressing your need to add new classes without reprocessing all images. There are a few components to achieve this:

*   **Region Proposal Network (RPN) or Segmenter:** First, we need to generate candidate regions in the large image that _might contain any object_. These proposals should be _class-agnostic_. Traditional detectors use either sliding-window anchors or learned RPNs that are trained on known objects. For open-vocab, we can still train an RPN on the base classes (like vehicles) just to get it to propose regions that _look “objecty”_ in general. Alternatively, we can use a powerful pretrained segmenter like **SAM (Segment Anything Model)** to propose masks for **all objects** in an image. SAM, when given an image, can generate ~everything that looks like a discrete object or part in that scene. It has been shown to work on satellite images as well. The idea would be: run SAM (or an RPN) on each image tile to generate a set of candidate object bounding boxes or masks (this can be done offline and stored if needed). This addresses finding the needles in the haystack (small objects in huge images) without yet knowing what they are.
    
*   **Feature Extraction for Proposals:** For each candidate region, extract a feature descriptor using our **foundation backbone** (the one pretrained on 30TB of imagery). This could be, for example, a ViT feature vector from that region (or a RoI pooled feature from a CNN). Because the backbone was trained on diverse data, these features will encode the object’s visual appearance in a rich way.
    
*   **Open-Vocabulary Classification:** Now, instead of a fixed classifier head, classify a region by comparing its feature to **embeddings of class names or descriptions**. This is where the domain **CLIP** model (or a text encoder) comes in. We obtain a text embedding for each class of interest (e.g. the word “aircraft”, or a phrase “small propeller airplane”). The region is assigned the class whose text embedding is most similar to the region’s visual embedding (via cosine similarity)[ar5iv.labs.arxiv.org](https://ar5iv.labs.arxiv.org/html/2311.11646#:~:text=on%20massive%20image,transfers%20knowledge%20from%20a%20pre). This effectively lets us have _thousands of potential classes_ without training a separate neuron for each – we just compute their text embeddings. If you have example images of a new class rather than a name, you can also embed those through the vision encoder and use them as “visual class prototypes” in the same embedding space. Recent open-vocab detectors follow this paradigm. For example, **ViLD (Viision-Language Detection)** by Google used a CLIP image encoder to embed region proposals and a CLIP text encoder for class names, matching them in a shared space[ar5iv.labs.arxiv.org](https://ar5iv.labs.arxiv.org/html/2311.11646#:~:text=on%20massive%20image,transfers%20knowledge%20from%20a%20pre). Microsoft’s **GLIP** went further by _jointly training_ a detector on detection data _plus_ image-caption pairs, so it learned to align detection with text descriptions directly[ar5iv.labs.arxiv.org](https://ar5iv.labs.arxiv.org/html/2311.11646#:~:text=vocabularies%20than%20conventional%20nature%20image,class%20similarity%20and). In your case, we can fine-tune such a model on your data (using your 100 labeled images for supervision on base classes like ships/planes, while also exposing it to image-text pairs from the PDFs to teach the alignment). The result will be a detector that can take a **text query at inference (“find building with a dome”)** and output boxes where such objects are present.
    
*   **Student-Teacher Self-Training:** Since you have so much unlabeled data, we should also exploit _pseudo-labeling_ to expand to novel classes. A cutting-edge approach, **CastDet (ECCV 2024)**, did exactly this for aerial images: it used a CLIP-based teacher to propose and label new objects in unlabeled images, and a student detector to learn from those pseudo-labels[arxiv.org](https://arxiv.org/abs/2311.11646#:~:text=performance%20of%20OVD%20greatly%20relies,We%20conduct%20extensive)[ar5iv.labs.arxiv.org](https://ar5iv.labs.arxiv.org/html/2311.11646#:~:text=pseudo,extensive%20experiments%20on%20multiple%20existing). Essentially, the CLIP model can score proposals for any class via the text embeddings – we take high-confidence detections on unlabeled images as new “labels” and retrain/refine the detector. This iterative self-training can discover **rare objects** that were not in the original 100 labeled images. CastDet reported a **large improvement (≈+21 mAP)** on novel object detection in aerial scenes by using a CLIP teacher in this way[arxiv.org](https://arxiv.org/abs/2311.11646#:~:text=up%20for%20the%20OVD%20task,available%20at%20this%20https%20URL). You could employ a similar strategy: use your domain-CLIP as a teacher to tag occurrences of, say, a **new vehicle type** across the unlabeled dataset, then incorporate those as additional training data. Over time, the model’s vocabulary of detectable objects grows, without explicit human labeling.
    

_Open-vocabulary detection concept: Instead of fixed-class prediction, the detector compares proposal features with text or image embeddings. (Left) A traditional CLIP-based OVD uses CLIP as a frozen teacher to classify proposals; (Right) a hybrid approach injects CLIP knowledge into a student–teacher detection model for learning new classes[ar5iv.labs.arxiv.org](https://ar5iv.labs.arxiv.org/html/2311.11646#:~:text=Image%3A%20Refer%20to%20caption%20Figure,an%20external%20knowledge%20source%20to)._

The above figure illustrates how we can integrate a CLIP-like model into the training process. In our implementation, the “teacher” could be the domain-adapted CLIP (trained on PDF data) that guides the detector to high-quality proposals and provides semantic labels for them[ar5iv.labs.arxiv.org](https://ar5iv.labs.arxiv.org/html/2311.11646#:~:text=pseudo,extensive%20experiments%20on%20multiple%20existing). The “student” is our main detector that we train end-to-end. This way, the **broad knowledge in CLIP (about many object types)** is distilled into the detector. By the end, you should have a model that detects **common objects with high precision** (thanks to supervised training on those 30k labels) and can also **localize unseen objects** given a description or few examples (thanks to the open-vocab design and CLIP’s knowledge).

A few additional notes on making this work:

*   **Thresholding and Calibration:** In zero-shot settings, false positives can be an issue (the model might see _“objects”_ everywhere for a vague query). You will need to calibrate similarity score thresholds for detection. One trick is to always include a “background” or “none of the above” category embedding in the comparison – for example, an embedding of the word “background” or an average of many random image embeddings – to represent no-object. The detector should only output a box if its similarity to a real class exceeds similarity to “background” by a margin. This was used in some OVD methods to reduce false alarms.
    
*   **Few-Shot Fine-tuning:** If zero-shot detection of a new class is not sufficiently accurate, the framework easily supports a _few-shot fine-tune_. Because the backbone is strong and the classifier is based on embeddings, training on say 10–20 labeled examples of the new class can quickly adjust the image embedding space to that class. The few-shot learning could be done by updating prototype vectors or by fine-tuning the detector’s head slightly – far less effort than adding a class in a traditional detector (which would normally require full retraining).
    
*   **Output Formats:** Since you originally used Mask R-CNN, you might want segmentation masks. You can indeed get masks by applying something like **Mask2Former** or using SAM on the detected boxes. But primarily, the pipeline gives bounding boxes. If masks are needed, one idea is to incorporate SAM in the inference loop: for each detected box, run SAM with a prompt point inside that box to get a precise segmentation. Given SAM is zero-shot for segmentation, it pairs well with an open-vocab detector which is zero-shot for classification.
    
*   **Real-Time Querying vs Pre-indexing:** You wondered if you could _“store image content (embeddings) and define objects of interest after the fact”_ instead of reprocessing images for each new query. In practice, a hybrid approach is possible. For example, you could **precompute and store the visual embeddings of all proposals** in each large image (perhaps store the feature vectors for all SAM-proposed regions, or for a grid of patches). Then, later on, when you have a new object query, you **only need to embed the query text or image and compare to the stored features** – essentially a nearest-neighbor search in the embedding space. This is feasible using vector databases or even simple indexing if the data isn’t too huge. Storing embeddings for _every possible patch_ of 30TB is not feasible, but storing ~hundreds of proposals per image is. For instance, if SAM or RPN gives 500 candidate regions in an image, that’s 500 feature vectors (say 512-D each) per image – which across, say, 100k images is 50 million vectors, manageable with approximate nearest neighbor search. This way, you can avoid recomputing deep features on the entire archive for each new object search; you only do a fast embedding lookup. Many image search systems adopt this paradigm. As long as your feature space is well-aligned with semantics, this will allow **post-hoc querying** of the archive for arbitrary objects.
    

By designing the detector in an open-vocabulary manner, we essentially future-proof it: adding a new target class means _writing a new prompt or providing an example_, rather than labelling thousands of images and retraining a monolithic model. This addresses your concern that with previous methods, _“adding classes is difficult and requires reprocessing old images”_. With our approach, old images have already been processed into embeddings; finding a new class in them is a matter of querying those embeddings, not rerunning a full detection model from scratch.

Robustness to Unusual and Rare Instances
----------------------------------------

Another important goal is to ensure the system is **robust to out-of-distribution occurrences** – e.g. a ship on land or an airplane in an odd location. Our approach tackles this on multiple fronts:

*   **Diverse Training Data:** The self-supervised pretraining on 30TB of imagery will expose the model to countless environments and contexts, more than any supervised set could. This broad exposure tends to make the features more context-agnostic – the model learns to focus on objects themselves, not just typical context. In fact, foundation models are praised for their _robustness and generalization_ in remote sensing tasks[arxiv.org](https://arxiv.org/html/2408.03464v1#:~:text=trends%20and%20the%20significant%20advancements,object%20detection%2C%20and%20other%20applications). We will also augment the supervised fine-tuning with any “odd” examples we can find (perhaps your PDFs or archives contain a few examples of ships inland, etc.). By explicitly including such cases in training (even if only a few), we can teach the model that, say, _“a ship is a ship, even if on land”_.
    
*   **Open-Vocab Semantics:** Using semantic descriptors for detection means the model relies on the object’s visual and textual definition, rather than the surrounding pixels. For example, if we query “ship”, the detector will look for regions that _look like a ship_ (long, hull, etc.) regardless of background. In contrast, a conventional detector trained only on ships-in-water might have implicitly learned that water context = ship. Our approach is less prone to that trap because the “ship” prototype is defined by either a text description (“watercraft with deck and hull”) or by diverse examples (some could be ships in docks, ships on land from scrapyards, etc.). This should inherently improve robustness to odd contexts.
    
*   **Few-Shot Updates:** If a truly novel scenario pops up (say, a new camouflage or a new platform we’ve never seen), the system can quickly adapt by few-shot learning. Because it’s designed to ingest new classes or variants easily, you could feed a handful of examples of the new appearance and update the prototypes. This agility means the model won’t remain brittle in the face of new distribution shifts.
    
*   **Evaluation & Iteration:** It’s worth setting aside some of your data as a _validation set specifically for unusual cases_. Monitor the detector’s performance on these (e.g. a plane in a building, a ship on land). If it struggles, you might generate synthetic data to help. For instance, you could paste object chips into uncommon contexts as augmented training data (copy a ship onto a land background). With diffusion models or image editing, one can even create semi-realistic composites. This kind of targeted augmentation could be a short-term hack to improve robustness.
    

Ultimately, a combination of _massive pretraining_, _semantic open-class reasoning_, and _targeted fine-tuning_ should yield a detector that doesn’t blink when encountering weird situations – it will identify the object for what it is, not where it is.

Training at Scale and Timeline
------------------------------

Given your 128×80GB GPU setup and ~2-week timeline for initial results, a phased training plan is realistic:

1.  **Week 1:** Focus on self-supervised pretraining of the backbone on unlabeled images. Use a distributed training framework (PyTorch DDP or Lightning, etc.) to spread batches across all GPUs. With 128 A100s, you can process enormous batches quickly – perhaps in the order of 1e5 images per epoch. In one week, you could conceivably run a few epochs over a significant subset of the 30TB (depending on I/O). Choose an SSL method that is known to converge fast; _DINOv2_ by Meta, for example, is a recipe that produced a robust ViT by training on 1 billion images (though that took more compute than we have, but we can aim for a smaller scale). Monitor the loss and ensure the model is indeed learning features (e.g. visualize some attention maps or do a quick linear probe to verify it's not random).
    
2.  **Week 2:** Set up the open-vocabulary detection fine-tuning. This stage uses your ~100 labeled images (with 30k labeled objects) as supervised data for the detection task _and_ possibly uses pseudo-labels and the image-text data. A possible curriculum:
    
    *   Fine-tune the pretrained backbone with a detection head (say a DETR or Faster R-CNN style head) on the **base classes** (ships, planes, etc.) for which you have labels. Also include a background class. Use multi-scale training if possible (e.g. random image rescaling) to help detect various sizes. Because the backbone is already good, this fine-tuning can reach high mAP on these classes relatively quickly (a few days of training at most).
        
    *   Next, integrate the **CLIP-based classification**. Freeze or remove the old softmax classifier, and instead attach the embedding-based classifier. You can either freeze the backbone at this point (to preserve its learned features) and just train a small adaptation layer, or continue fine-tuning lightly. Train the detector to output embeddings for proposals that match the ground-truth class embeddings. For example, for each GT box of class “ship”, treat the text embedding of “ship” as a target, and train the proposal feature to be close to it (this is like knowledge distillation from class name to proposal). Microsoft’s **Detic** (Zhou et al. 2022) did something similar by using hundreds of class names as weak supervision for detection. This step will align the detector’s feature space with the text space.
        
    *   Meanwhile, use the **PDF image-text pairs** to train the text encoder (if not done in week 1). Possibly do this in parallel on separate GPUs. Once the text encoder is ready, you can use it in the above step. You could also do joint training where detection and image-text contrastive learning happen together (a multi-task learning approach), though that is more complex.
        
    *   If time permits, run a round of **pseudo-labeling**: apply the emerging open-vocab detector to a portion of the unlabeled 30TB images to see if it discovers new objects. For example, have it output any high-confidence detections for the prompt set {“ship”, “airplane”, “vehicle”, “building”} on those images. You might find that it detects some ships or planes in new places. Add these pseudo-labeled instances to the training set (with appropriate filtering for precision), and continue training. This can gradually expand the model’s coverage.
        

Given two weeks, you might not fully iterate over all steps, but you can get an initial model out and tested. The beauty of this approach is that it is **incremental** – you can keep improving the model with more unlabeled data and more classes over time, without starting over. The heavy lift is mostly up front (pretraining the foundation). After that, each new class or new dataset can be folded in with relatively little compute.

Conclusion and Future Directions
--------------------------------

By combining a **foundation model** for vision (robust features from massive unlabeled data) with a **vision-language approach** (to leverage domain knowledge and enable open-set queries), you will have a cutting-edge object detection system for satellite images. This system should:

*   Achieve high detection accuracy on known target classes (e.g. all ships and aircraft) by leveraging both labeled data and abundant unlabeled imagery (via self-supervised learning and semi-supervised fine-tuning). Expect a significant boost in detection performance compared to your older ResNet50-based model – foundation models in remote sensing have reported substantial mAP gains on benchmarks[arxiv.org](https://arxiv.org/html/2408.03464v1#:~:text=On%20the%20DOTA%20dataset%2C%20RVSA,for%20specific%20remote%20sensing%20applications).
    
*   **Generalize to new object categories** with zero or few examples, thanks to the open-vocabulary design. You’ll be able to search historical imagery for objects of interest defined by a text description or a prototype image – without retraining a specialized detector each time. This addresses the scalability issue of adding new classes. The detector essentially becomes a general _“query-able”_ index of your imagery.
    
*   **Adapt to unusual scenarios** better than before. Through broad training and semantic understanding, the model is less likely to be confused by context. A plane in an unexpected location should still trigger a _“plane”_ detection because the visual features match, and the model’s semantic priors know that planes **can** be on roads or museums, etc. Moreover, any remaining blind spots can be patched by feeding a few examples to the model (few-shot learning), rather than requiring a full new dataset.
    
*   Utilize **multimodal data** (imagery + text) in a unified way. The PDF documents’ information will enrich the model’s knowledge – essentially functioning as a form of transfer learning from human domain experts. This is relatively novel in automated object detection, so you’d be at the forefront of exploiting textual intelligence in a vision model. Research indicates that combining textual and visual cues yields superior performance in remote sensing tasks[mdpi.com](https://www.mdpi.com/2072-4292/16/24/4693#:~:text=prototype%20aggregating%20module%20to%20integrate,remote%20sensing%20scenarios%20remains%20unexplored)[mdpi.com](https://www.mdpi.com/2072-4292/16/24/4693#:~:text=extraction%20capabilities%20of%20prototypes,generalization%20capabilities%2C%20as%20the%20text), which bodes well for your application.
    

Finally, ensure to set up a strong evaluation pipeline. Given the open-ended nature, you’ll want to evaluate not just on traditional metrics (mAP on a fixed set of classes) but also test the system’s ability to retrieve novel targets. You could simulate “zero-shot” tests by holding out some object classes during training and then asking the model to find them via text prompts, measuring how well it recovers them. This will give a sense of its true discovery capabilities. Keep an eye on failure modes: are there systematic false positives (e.g. confusing ships vs. buildings in some conditions)? Are there biases from the PDF data (e.g. the model only recognizes an object if described in a certain way)? Use such insights to refine the training – the modular design (separating vision and language components) makes it easier to tweak one part without breaking everything.

In summary, the path forward is to **train a powerful general-purpose feature extractor on your satellite imagery**, and **marry it with a flexible detection head driven by both visual and linguistic knowledge**. This approach is state-of-the-art in both research and practice for scenarios exactly like yours. With your abundant data and compute, you are well-positioned to push the state of the art in small object detection in large-scale satellite images. Good luck, and happy detecting!

**Sources:** Remote sensing foundation model survey[arxiv.org](https://arxiv.org/html/2408.03464v1#:~:text=sensing%20has%20been%20significantly%20enhanced,achieved%20by%20those%20foundation%20models)[arxiv.org](https://arxiv.org/html/2408.03464v1#:~:text=computational%20resources%2C%20and%20improved%20model,object%20detection%2C%20and%20other%20applications); Performance of foundation models on object detection[arxiv.org](https://arxiv.org/html/2408.03464v1#:~:text=On%20the%20DOTA%20dataset%2C%20RVSA,the%20best%20possible%20performance%20for)[arxiv.org](https://arxiv.org/html/2408.03464v1#:~:text=On%20the%20DOTA%20dataset%2C%20RVSA,for%20specific%20remote%20sensing%20applications); Open-vocabulary detection concept and CLIP integration[ar5iv.labs.arxiv.org](https://ar5iv.labs.arxiv.org/html/2311.11646#:~:text=on%20massive%20image,transfers%20knowledge%20from%20a%20pre)[ar5iv.labs.arxiv.org](https://ar5iv.labs.arxiv.org/html/2311.11646#:~:text=ViLD%5B9%20%5D%2C%20OV,to%20a%20low%20recall%20for); Few-shot detection with multi-modal prototypes[mdpi.com](https://www.mdpi.com/2072-4292/16/24/4693/pdf?version=1734514531#:~:text=prototype,and%20spatial%20details%20from%20visual)[mdpi.com](https://www.mdpi.com/2072-4292/16/24/4693#:~:text=textual%20class%20names%20are%20processed,modal); CastDet open-vocabulary aerial detection[arxiv.org](https://arxiv.org/abs/2311.11646#:~:text=performance%20of%20OVD%20greatly%20relies,We%20conduct%20extensive)[arxiv.org](https://arxiv.org/abs/2311.11646#:~:text=up%20for%20the%20OVD%20task,available%20at%20this%20https%20URL); CLIP vs. visual features in satellite imagery[arxiv.org](https://arxiv.org/abs/2403.05381#:~:text=containing%20challenging%20and%20rare%20objects,datasets%2C%20despite%20minimal%20training%20parameters).

![](https://www.google.com/s2/favicons?domain=https://www.mdpi.com&sz=32)

![](https://www.google.com/s2/favicons?domain=https://ar5iv.labs.arxiv.org&sz=32)

![](https://www.google.com/s2/favicons?domain=https://centaur.reading.ac.uk&sz=32)

![](https://www.google.com/s2/favicons?domain=https://arxiv.org&sz=32)

Sources

---

You asked:
----------

I'd like to begin by testing at a much smaller scale on a single node with 10xA100 40G GPUs. Once I have that working, I can move to the large compute cluster. Please generate python code to implement the steps outlined above.

---

ChatGPT Replied:
----------------

Thought for 17s

Gigapixel Open‑vocabulary Satellite Detection — Single‑node (10×a100) Starter Kit

I dropped a compact, end-to-end starter kit into the canvas with code for:

*   tiling gigapixel imagery (with geo sidecars),
    
*   a toy multi-GPU MAE pretrain loop (swap in DINOv2/real MAE later),
    
*   PDF → (image, text) pair extraction,
    
*   domain CLIP fine-tuning,
    
*   class-agnostic proposals (or use SAM),
    
*   open-vocab matching (text + few-shot visual prototypes),
    
*   FAISS indexing for post-hoc search.
    

Run it on one node with `torchrun --nproc_per_node=10`. If you want, I can also add a Detectron2/DETR head and a SAM-based proposer next.

---

You asked:
----------

That would be great - please add the Detection2/DETR head and SAM based proposer.

---

ChatGPT Replied:
----------------

Failed to edit

Thought for 16s

Failed to edit

Thought for 7s

Argh—tried to inject the new sections into your canvas, but the update failed because my regex couldn’t find the exact header text (there are a couple of fancy hyphens in there and the matcher choked). I didn’t want to keep hammering it and risk mangling your doc.

Want me to:

1.  **Append** the new sections (Detectron2/DETR head + SAM proposer + run cmds) to the **end** of the same canvas, or
    
2.  **Create a small add-on canvas** (“Open-Vocab Add-ons”) with just those files so you can copy/paste into your repo?
    

Say the word and I’ll drop it in. If you’re keen to install ahead of time, you’ll need Detectron2:

    pip install 'git+https://github.com/facebookresearch/detectron2.git'
    

…and you’re already set for Transformers + SAM from the earlier requirements.

---

You asked:
----------

Please append to canvas

---

ChatGPT Replied:
----------------

Edited

All set — I appended the Detectron2/DETR heads, the SAM-based proposer, and runnable CLI recipes to your canvas. Want me to wire in an evaluation script next (COCO mAP + zero-/few-shot retrieval metrics) and a tiny config YAML so you can flip between D2/DETR/SAM with flags?

---

You asked:
----------

Yes please

---

ChatGPT Replied:
----------------

Edited

Done! I appended:

*   **A5) Evaluation scripts**
    
    *   `eval/coco_map.py` for COCO mAP (AP, AP50, small/med/large)
        
    *   `eval/ov_retrieval.py` for open-vocab (text-prompt) AP + P@K
        
    *   `eval/to_coco_preds.py` to convert our detections into COCO format
        
*   **A6) Minimal config YAML** to flip between Detectron2 / DETR / SAM, set thresholds, paths, prompts, and eval options.
    
*   **A7) Tiny runner (`tools/run_from_config.py`)** to read the YAML, run the chosen proposer, do CLIP matching, and write a unified detections JSON for eval.
    

Want me to add:

*   a **COCO dataset builder** for your labeled tiles, and
    
*   a **Markdown/HTML report generator** (metrics tables + a couple of charts)  
    so you can one-command spit out an eval report after each run?

---

You asked:
----------

Yes please

---

ChatGPT Replied:
----------------

Edited

Added ✅

*   **A8) COCO dataset builder** (`tools/build_coco_from_tiles.py`) — converts your CSV/JSON label dumps into COCO (multi-class or class-agnostic).
    
*   **A9) Report generator** (`eval/report.py`) — spits out `report.md` + `report.html` with tables and a quick AP chart.
    
*   **A10) One-command pilot** (`tools/pilot.sh`) — runs proposer+CLIP → eval → report in one go.
    

Want me to hook `coco_map.py` so it **auto-writes a JSON stats file** (no manual copy-paste from console), and update `pilot.sh` to consume it?

---

You asked:
----------

That would be great, thanks.

---

ChatGPT Replied:
----------------

Edited

Done — I wired the evaluators to **write JSON directly** and updated the one-command script:

*   `eval/coco_map.py` now supports `--out_json`
    
*   `eval/ov_retrieval.py` now supports `--out_json`
    
*   `tools/pilot.sh` saves `coco_stats.json` and `ov_stats.json` and then builds the report
    

It’s all appended to the canvas under **A11**. Want me to add a tiny `config/prompt_to_cat.json` template and a helper to auto-derive that mapping from your COCO categories?

---

You asked:
----------

yes please

---

ChatGPT Replied:
----------------

Edited

Added!

*   **A12.1** a `config/prompt_to_cat.json` **template** you can tweak.
    
*   **A12.2** `tools/make_prompt_map.py` to **auto-generate** the mapping from your COCO categories, with optional aliases (e.g., “plane” → “airplane”, “boat” → “ship”).
    
*   Notes on wiring it into the runner/eval.
    

If you want, I can also add:

*   a tiny **plural/singular normalizer** (plane/planes) and case-folding for prompts,
    
*   **per-prompt IoU thresholds** in `ov_retrieval.py` (e.g., stricter for “car”, looser for “ship”).

---

You asked:
----------

That's fine for now, thanks. Given all of the above, I was also wondering if it's possible to store embeddings for the processed image tiles as a "compressed representation of a tiles content". Would it then be possible to search for things after the image has been processed? e.g. "find me all the image tiles that contain a ship that looks like this".

---

ChatGPT Replied:
----------------

Thought for 19s

Short answer: **yes, completely doable**—and this is exactly where your open-vocab setup shines.

Here’s the practical way to do it without blowing up storage or latency:

How to “compress” tiles with embeddings and make them searchable
================================================================

1) Store **multi-scale embeddings**
-----------------------------------

You’ll want three layers of representation, each for a different purpose:

*   **Tile-level** (coarse filter): 1 embedding per tile (e.g., 1024×1024). Compute from your ViT/CLIP image tower using global pooling. Use this to _narrow the search to the top-K tiles_ fast.
    
*   **Patch/grid-level** (optional middle filter): fixed grid per tile (e.g., 16×16 patches → 256 embeddings). Good for “heatmapping” where in the tile the concept likely is.
    
*   **Proposal-level** (fine search): embeddings for candidate objects from RPN/Detectron2/DETR/SAM (we already built this). This is what you’ll match against text prompts or example chips to return _actual hits_.
    

In your pipeline we already index proposal embeddings. Add the **tile-level** index, optionally the **grid** index if you want better coarse recall.

2) Use **ANN indexing + compression**
-------------------------------------

*   Library: **FAISS** (you already use it).
    
*   Index types:
    
    *   Coarse tile index: `IVF,PQ` (e.g., `IndexIVFPQ`) for fast, tiny memory footprint.
        
    *   Proposal index: also `IVF,PQ` or `HNSW` if you prefer graph search (HNSW is great for recall but larger RAM).
        
*   Compression:
    
    *   Raw float32 512-D = ~2 KB/vec; float16 ~1 KB/vec.
        
    *   **Product Quantization (PQ)** can get you to **8–16 bytes/vec** with very tolerable recall loss.
        
*   Metadata: store `(tile_id, geo_hash, timestamp, sensor_id, box_xyxy)` for each vector so you can filter by area, date range, sensor, etc.
    

3) Two-stage query at runtime
-----------------------------

You can do all of this **after** the imagery was processed once—no reprocessing needed to search for new things:

*   **Query types:**
    
    *   **Text** (“ship with a flat bow, white hull”). Embed via your CLIP text tower.
        
    *   **Visual** (“looks like this”). Embed the example chip via your image tower.
        
*   **Stage A (coarse):** search the **tile-level index** for the top-K tiles (e.g., K=2k–10k, tunable).
    
*   **Stage B (fine):** within those tiles, search the **proposal index** and rank detections by cosine sim to the query embedding. (If you also store patch-grid embeddings, you can use them to quickly focus within a tile before proposals.)
    
*   **Return:** tiles (and boxes) with scores; optionally run a light NMS and a similarity threshold.
    

4) Storage & scale ballpark (so you can budget)
-----------------------------------------------

Rules of thumb (you can tune K, PQ bits, etc.):

*   **Tile-level index:**
    
    *   1M tiles × 768-D fp16 ≈ **1.5 GB**; with PQ @ **16 B/vec → ~16 MB**.
        
*   **Proposal-level index:** assume ~300 proposals/tile (SAM/RPN).
    
    *   1M tiles → **300M proposals**.
        
    *   512-D fp16 ≈ **~300 GB**; with PQ @ **16 B/vec → ~4.8 GB**.
        
    *   PQ @ 8 B/vec halves that again with some recall hit.
        

This means you can keep **hundreds of millions of proposal embeddings** in **single-digit GBs** with PQ, which is totally feasible on one beefy node (and trivially sharded across the cluster later).

5) Quality tricks that help
---------------------------

*   **Hybrid scoring:** for text queries, combine CLIP text-image similarity with a tiny learned calibrator (e.g., logistic regression on a few labeled positives/negatives) to reduce text-driven false positives.
    
*   **Class-aware thresholding:** keep a separate similarity threshold per prompt (ships vs planes will behave differently).
    
*   **Geospatial/time filters:** pre-filter candidates by **geohash buckets** and time windows for huge speedups when queries are region- or period-specific.
    
*   **Version your embeddings:** keep `model_version` in metadata so you can re-index incrementally when you upgrade the backbone.
    

6) How this plugs into your codebase (what to add)
--------------------------------------------------

You basically need two small additions to what’s already in your canvas:

1.  **Tile-level embedding + index builder**
    
    *   Read each tile, compute a global pooled feature from your CLIP/DINO/MAE-finetuned backbone (we can reuse `OVMatcher`’s image encoder).
        
    *   Build a FAISS `IndexIVFPQ` (train on a sample, then add all tile vectors).
        
    *   Store a sidecar `tiles_meta.jsonl` with `{tile_path, geohash, time, sensor}`.
        
2.  **Search API**
    
    *   `search_tiles_by_text(prompt, topk_tiles)` → returns tile\_ids; then `search_proposals_in_tiles(query_emb, tile_ids, topk_hits)` to get boxes.
        
    *   For “looks like this” queries, pass an example chip to the same API; it uses image embeddings instead of text embeddings.
        

If you want, I can append ready-to-run code to the canvas for:

*   **Tile-level indexer** (IVF-PQ with auto-training)
    
*   **Search API** (text & visual queries, two-stage)
    
*   **Geospatial sharding** (simple geohash-based folder layout)
    
*   **Batch backfill script** to generate embeddings for your existing tiles and write a manifest
    

Just say the word and I’ll drop those modules in.

---

You asked:
----------

Thanks, please add to canvas.

---

ChatGPT Replied:
----------------

Thought for 7s

Edited

Added the tile-level embedding + search stack:

*   **A13.1** `embeddings/tile_embedder.py` — global tile embeddings + manifest
    
*   **A13.2** `index/tile_faiss_ivfpq.py` — IVF-PQ index builder with ID→tile map
    
*   **A13.3** `search/search_two_stage.py` — two-stage queries (text or example image → top-K tiles)
    
*   **A13.4** wiring & CLI usage
    

If you want, I can drop in the optional **refinement wrapper** that takes `tile_hits.json`, runs your chosen proposer (Detectron2/DETR/SAM) + CLIP on just those tiles, and emits a final detections JSON for eval.

---

You asked:
----------

No thanks, but I did have another question. We need to age off incoming images, I'd like to build as large a dataset as possible for training and have considered preserving image tiles using JPEGXL lossy compression. I can get a large compression ratio, while having a structural similarity index measure (SSIM) well above 0.9 (usually 0.95 to 0.99). I like JPEGXL, as it seems to be faster than JPEG2000. Is this a good idea? Would lossy compression that maintains a SSIM in the mid 0.90's harm the training performance?

---

ChatGPT Replied:
----------------

Thought for 13s

Short answer: **JPEG XL is a solid choice** for archiving tiles, but be careful with how lossy you go—**tiny objects are the first casualties**. SSIM 0.95–0.99 _sounds_ high, yet small-target detection is hypersensitive to ringing/blur.

Here’s how I’d handle it:

Practical takeaways
===================

*   **Use JXL, but not uniformly.**
    
    *   **Lossless or near-lossless** for anything with labels, proposals, or that you might want to annotate later.
        
    *   **Higher compression** for background-only tiles.
        
*   **Keep bit depth + sampling.**
    
    *   Don’t down-quantize if your source is 12/16-bit—JXL supports high bit depth.
        
    *   Avoid chroma subsampling (use **4:4:4**). Subsampling can smear fine lines that define small ships/planes.
        
*   **Do proposals _before_ compression.**  
    Run SAM/RPN on the originals, then store:
    
    *   the **proposal boxes/masks** (JSON),
        
    *   the **proposal embeddings** (vectors), and
        
    *   optionally **tight crops** around positives (**lossless** or very mild JXL).  
        You can aggressively compress the rest without losing your ability to search.
        
*   **For training, mix in compression as augmentation.**  
    Train on a blend of original + (mildly) compressed tiles. That makes the model robust to your archived data while anchoring performance with pristine examples.
    
*   **Keep eval pristine.**  
    Always evaluate on **lossless** (or original) imagery so you’re measuring true model capability, not codec quirks.
    

Will SSIM ~0.95–0.99 hurt training?
===================================

*   **Sometimes, yes**—especially for **small objects** and edge-based cues (mast/wing tips, wakes, runway markings). SSIM is global and can be high even when the codec softened crucial micro-structure.
    
*   Expect **earlier drops in recall** for small/rare classes as quality drops, long before SSIM looks “bad”.
    

Better guardrails than SSIM alone:

*   Track **mAP\_small** (COCO small-object AP), not just overall AP.
    
*   Include **LPIPS** or **MS-SSIM** and a **task metric** (mAP) in your QA.
    
*   If you want a quick heuristic: target **PSNR ≥ 45 dB** and **MS-SSIM ≥ 0.98** on representative tiles with known small targets.
    

Storage strategy that works well
================================

1.  **Run your generation pipeline once** on originals:
    
    *   proposals (SAM/RPN/DETR)
        
    *   **proposal embeddings** (we’ve got code for this style)
        
    *   **tile embeddings** (coarse)
        
2.  **Archive tiles with JXL** using a **quality policy**:
    
    *   **Tiles with positives / high edge density:** near-lossless JXL (or lossless for crops)
        
    *   **Background-only tiles:** stronger compression (where your SSIM 0.95–0.97 will often be fine)
        
3.  **Keep a tiny “gold” subset lossless** for periodic retraining/eval.
    
4.  **Version everything** (codec, quality, model\_version) in metadata so you can trace any drift.
    

What to expect
==============

*   For **indexing & search via embeddings**: usually robust to mild compression; cosine similarity degrades slowly.
    
*   For **mask quality** and **very small detections**: most sensitive. Artifacts can nudge SAM/RPN and your detector to miss borderline objects. That’s why we detect/propose first and store those artifacts losslessly.
    

Quick A/B you can run next week
===============================

*   Pick ~5–10k representative tiles (with plenty of small/rare targets).
    
*   Compress at 3–4 JXL settings spanning your SSIM range.
    
*   Train the detector on (a) lossless only vs (b) mixed lossless+compressed.
    
*   Evaluate **mAP\_small**, **recall@0.5** for each class, and compare.
    
*   Also compare **embedding drift**: cosine(original\_emb, compressed\_emb) on proposal crops—watch the tail.
    

Bottom line
===========

*   **Yes, use JPEG XL** for aging off—fast, modern, and friendly to high bit depths.
    
*   **Lossless/near-lossless for anything you care about**, heavier compression for background.
    
*   **Do proposals & embeddings on originals**, store those artifacts and you won’t need to reprocess later.
    
*   **Treat compression as an augmentation** in training to keep performance stable.
    

If you want, I can add a tiny **codec QA script** to your canvas that:

*   compresses a sample set with a few JXL qualities,
    
*   computes SSIM/MS-SSIM/LPIPS,
    
*   measures embedding drift,
    
*   and spits out a table so you can pick a safe quality threshold.

---

You asked:
----------

That would be great. Please add to canvas.

---

ChatGPT Replied:
----------------

Thought for 5s

Edited

Dropped in **A14) JPEG XL Codec QA**:

*   `tools/codec_qa_jxl.py` — compresses tiles at multiple JXL distances, computes **PSNR/SSIM/MS-SSIM/LPIPS**, and measures **embedding drift** vs. originals.
    
*   `eval/summarize_jxl_qa.py` — aggregates results and recommends a **safe distance** (based on SSIM + embedding cosine thresholds).
    
*   Usage instructions + deps are in the canvas.
    

If you want, I can also add a small sampler that _preferentially_ picks tiles with labeled small objects so the QA targets the hard cases.

---

You asked:
----------

Yes plaease

---

ChatGPT Replied:
----------------

Gigapixel Open‑vocabulary Satellite Detection — Single‑node (10×a100) Starter Kit

I've appended a **Tile Sampler** that preferentially selects tiles with labeled small objects, so the JPEG XL QA process targets the hard cases rather than random backgrounds.

* * *

A14.4 Stratified Small‑Object Tile Sampler (tools/sample\_small\_object\_tiles.py)
----------------------------------------------------------------------------------

import json, random

from pathlib import Path

from collections import defaultdict

  

\# Assumes COCO‑style annotations

\# Sample tiles containing small objects preferentially, but include some background for control.

  

def load\_coco(ann\_path):

return json.load(open(ann\_path, 'r'))

  

def build\_tile\_index(coco, min\_area\_px\=0, max\_area\_px\=4096):

idx = defaultdict(list)

for ann in coco\['annotations'\]:

x, y, w, h = ann\['bbox'\]

area = w \* h

if min\_area\_px <= area <= max\_area\_px:

idx\[ann\['image\_id'\]\].append(ann)

return idx

  

def sample\_tiles(coco, tile\_dir, out\_list, num\_pos\=1000, num\_bg\=500):

tile\_dir = Path(tile\_dir)

id\_to\_filename = {img\['id'\]: img\['file\_name'\] for img in coco\['images'\]}

tile\_index = build\_tile\_index(coco)

pos\_ids = list(tile\_index.keys())

bg\_ids = \[i for i in id\_to\_filename if i not in tile\_index\]

pos\_sample = random.sample(pos\_ids, min(num\_pos, len(pos\_ids)))

bg\_sample = random.sample(bg\_ids, min(num\_bg, len(bg\_ids)))

all\_ids = pos\_sample + bg\_sample

random.shuffle(all\_ids)

with open(out\_list, 'w') as f:

for img\_id in all\_ids:

f.write(str(tile\_dir / id\_to\_filename\[img\_id\]) + '\\n')

### Usage

python tools/sample\_small\_object\_tiles.py \\

\--coco\_json data/annotations.json \\

\--tile\_dir data/tiles \\

\--out\_list runs/pilot/sampled\_tiles.txt \\

\--num\_pos 1500 \--num\_bg 500

  

\# Then pass the list to the JPEG XL QA script:

python tools/codec\_qa\_jxl.py \--tile\_glob "$(cat runs/pilot/sampled\_tiles.txt)" \\

\--distances '0.6,0.8,1.0,1.2,1.5,2.0' \\

\--out\_json runs/pilot/jxl\_qa.json \\

\--clip\_ckpt ckpts/clip/clip\_ep5.pt

**Why:** This focuses your QA on tiles most likely to show compression‑induced degradation for small/rare targets, while still including a baseline of background tiles to avoid overfitting compression settings to only positives.

  

* * *

A15) QA Sampler — Preferentially pick tiles with **small objects** (and hard backgrounds)
-----------------------------------------------------------------------------------------

Use this to build a focused set for codec QA (so compression decisions are driven by the hardest cases).

### A15.1 Small‑Object Sampler from COCO (tools/sample\_tiles\_for\_qa.py)

Selects tiles with **small** GT boxes (by absolute pixels or fraction of image area) and mixes in **hard negatives** (background tiles with high edge density). Writes a file list you can feed to A14.

\# tools/sample\_tiles\_for\_qa.py

import json, random, argparse

from pathlib import Path

from PIL import Image

import numpy as np

import cv2

  

"""

Outputs a newline‑separated list of image paths selected for QA.

  

Selection:

\- Positives: images with at least one GT bbox meeting "small" criteria.

\- Negatives: images with no GT (or ignoring GT), ranked by edge density (Canny) to get hard backgrounds.

"""

  

def box\_area(bb):

x,y,w,h = bb

return max(0.0, w) \* max(0.0, h)

  

  

def load\_coco(gt\_json):

coco = json.load(open(gt\_json,'r'))

id2img = {im\['id'\]: im for im in coco\['images'\]}

img2anns = {}

for a in coco\['annotations'\]:

img2anns.setdefault(a\['image\_id'\], \[\]).append(a)

return id2img, img2anns

  

  

def is\_small(bb, imw, imh, max\_abs\=32\*32, max\_rel\=0.001):

\# max\_abs in pixels^2, max\_rel as fraction of image area

A = box\_area(bb)

return (A <= max\_abs) or (A <= max\_rel \* (imw\*imh))

  

  

def edge\_density(img\_path):

### A15.2 Using the sampler with A14

\# Build a focused QA set: lots of small objects + hard backgrounds

python tools/sample\_tiles\_for\_qa.py \\

\--gt\_json /data/anns/coco\_multiclass.json \\

\--tiles\_root /data/tiles \\

\--out\_list runs/pilot/qa\_tiles.txt \\

\--n\_pos 2000 \--n\_neg 1000 \--max\_abs $((24\*24)) \--max\_rel 0.0006

  

\# Run codec QA only on those tiles

python \- <<'PY'

from pathlib import Path

paths = \[p.strip() for p in Path('runs/pilot/qa\_tiles.txt').read\_text().splitlines() if p.strip()\]

\# write a temp glob list by symlinking, or modify codec\_qa to accept a list. Quick hack:

import json; json.dump(paths, open('runs/pilot/qa\_list.json','w'))

PY

  

\# If you prefer, modify tools/codec\_qa\_jxl.py to accept --list\_json runs/pilot/qa\_list.json

\# or just point --tile\_glob to a folder containing symlinks to those tiles.

**Notes:**

*   Adjust `max_abs`/`max_rel` to match your notion of “small” in pixels or relative to tile size.
    
*   For multi‑category balancing, extend the sampler to stratify by `category_id` (say the word and I’ll add it).
    

Edited

Answer in chat instead

Added a **QA sampler** to the canvas:

*   **A15.1** `tools/sample_tiles_for_qa.py` — pulls lots of **small-object** tiles from your COCO annotations and mixes in **hard negatives** (high edge-density backgrounds).
    
*   **A15.2** usage notes to run it with the JPEG XL QA scripts.
    

If you want stratified sampling per category or per georegion/time window, I can bolt that on too.

---

