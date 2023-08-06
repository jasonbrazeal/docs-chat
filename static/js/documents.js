const MAX_FILESIZE_MB = 100;

function validateFiles(files) {
  let sizeErrors = false;
  let typeErrors = false;
  let otherErrors = false;
  let problemFiles = [];
  const existingDocumentNodes = document.querySelectorAll(".document");
  const existingDocuments = Array.from(existingDocumentNodes).map(doc => doc.innerHTML);
  console.log('validating files:');
  for (file of files) {
    console.log(file);
    const fileName = file.name;
    const fileSize = file.size;
    const fileType = file.type;
    const sizeInMB = Number.parseFloat(fileSize / (1024 * 1024)).toFixed(2);

    console.log(fileName);
    console.log(fileSize);
    console.log(fileType);
    console.log(sizeInMB);

    if (sizeInMB > MAX_FILESIZE_MB) {
      sizeErrors = true;
      problemFiles.push(fileName);
      console.log(`${fileName} is too large: ${sizeInMB}MB`);
    }
    if (fileType !== "application/pdf") {
      typeErrors = true;
      if (!problemFiles.includes(fileName)) {
        problemFiles.push(fileName);
        console.log(`fileType is unacceptable: ${fileType || "cannot determine"}`);
      }
    }
    if (existingDocuments.includes(fileName)) {
      otherErrors = true;
      problemFiles.push(fileName);
      console.log(`${fileName} has already been uploaded`);
    }

  }

  if (!sizeErrors && !typeErrors && !otherErrors) {
    return ""
  }

  const fileSizeErrorMessage = `Individual pdfs must be < ${MAX_FILESIZE_MB}MB in size.`;
  const fileTypeErrorMessage = "Only pdf files are supported at this time.";
  const fileNameErrorMessage = "File name already exists.";

  let errorMessage = "";
  if (problemFiles) {
    errorMessage = `Problem uploading: ${problemFiles.join(", ")}` + errorMessage

    if (sizeErrors) {
      errorMessage += "\n" + fileSizeErrorMessage;
    }
    if (typeErrors) {
      errorMessage += "\n" + fileTypeErrorMessage;
    }
    if (otherErrors) {
      errorMessage += "\n" + fileNameErrorMessage;
    }
  }

  return errorMessage

}



function dropHandler(ev) {
  console.log("File(s) dropped");

  // Prevent default behavior (Prevent file from being opened)
  ev.preventDefault();
  ev.stopPropagation();

  [...ev.dataTransfer.files].forEach((file, i) => {
    console.log(`… file[${i}].name = ${file.name}`);
  });
  const form = document.getElementById('document-form');
  const input = document.querySelectorAll('#document-form input')[0];
  input.files = ev.dataTransfer.files;
  const error = validateFiles(input.files);
  if (error) {
    M.toast({text: error, displayLength: 10000});
    input.files = null;
  } else {
    const loader = document.getElementById("loader");
    loader.classList.remove("hide");
    form.submit();
  }
}

function dragOnHandler(ev) {
  console.log("File(s) on drop zone");
  ev.preventDefault();
  ev.stopPropagation();
  const dropZone = document.getElementById('drop-zone');
  dropZone.classList.add('drop-zone-dragging')
}

function dragOffHandler(ev) {
  console.log("File(s) off drop zone");
  ev.preventDefault();
  ev.stopPropagation();
  const dropZone = document.getElementById('drop-zone');
  dropZone.classList.remove('drop-zone-dragging')
}


document.addEventListener('DOMContentLoaded', function() {
  // initialize all modals
  const allModalElems = document.querySelectorAll('.modal')
  const allModals = M.Modal.init(allModalElems, {});

  // document upload
  const dropZone = document.getElementById('drop-zone');
  if (dropZone) {
    const events = ['drag', 'dragstart', 'dragend', 'dragover', 'dragenter', 'dragleave', 'drop'];
    for (const event of events) {
      dropZone.addEventListener(event, (e) => {
        e.preventDefault();
        e.stopPropagation();
      });
      dropZone.addEventListener('dragover', dragOnHandler);
      dropZone.addEventListener('dragenter', dragOnHandler);

      dropZone.addEventListener('dragleave', dragOffHandler);
      dropZone.addEventListener('dragend', dragOffHandler);
      dropZone.addEventListener('drop', dragOffHandler);

      dropZone.addEventListener('drop', dropHandler);
    }
    const form = document.getElementById('document-form');
    const button = document.querySelectorAll('#document-form button')[0];
    const input = document.querySelectorAll('#document-form input')[0];
    button.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      input.click();
    });
    input.addEventListener('change', (e) => {
      e.preventDefault();
      e.stopPropagation();
      const error = validateFiles(input.files);
      if (error) {
        M.toast({text: error, displayLength: 10000});
        input.files = null;
      } else {
        const loader = document.getElementById("loader");
        loader.classList.remove("hide");
        form.submit();
      }
    });
  }

  const documentForm = document.getElementById('document-form');
  if (documentForm) {
    htmx.on('#document-form', 'htmx:xhr:progress', function(evt) {
        htmx.find('#progress').setAttribute('value', evt.detail.loaded/evt.detail.total * 100)
      });
  }

  const clearDocsSubmit = document.getElementById('clear-docs-submit');
  clearDocsSubmit.addEventListener('click', () => {
    const clearDocsForm = document.getElementById('clear-docs-form');
    clearDocsForm.submit();
  });

});
