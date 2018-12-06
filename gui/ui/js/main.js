function change_checkbox(elem) {
    var checkbox = elem.parentElement.querySelector('.custom_checkbox_input');
    checkbox.checked = checkbox.checked ? false : true;
}

function start_script(elem, event) {
    var data = null, test = null, cls = null, cls_name = null, bagging = false;

    var command = "python3 ../classifier.py";


    bagging = document.querySelector('#bagging').checked;

    if (document.querySelector('#data').value.trim()) {
        command = `${command} -d "${ document.querySelector('#data').value }"`;
        
        if (document.querySelector('#test').value.trim()) {
            command = `${command} -t "${ document.getElementById("data").value.trim() }"`;
        }
    }

    if (bagging)
        command = `${command} -b`;

    cls = document.querySelector('#cls');
    cls_name = cls.options[cls.options.selectedIndex].innerHTML.trim();
    command = `${command} -c ${cls_name}`;

    eel.start_script(command)
}
